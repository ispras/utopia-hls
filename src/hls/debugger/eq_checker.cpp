//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "HIL/Ops.h"
#include "HIL/Utils.h"
#include "hls/debugger/eq_checker.h"

#include <iostream>

namespace eda::hls::eqchecker {

std::unique_ptr<EqChecker> EqChecker::instance = nullptr;

bool EqChecker::equivalent(Model &lhs, Model &rhs) const {

  Context ctx;
  ExprVector nodes(ctx);

  OptionalGraph lGraphOpt = mlir::hil::getGraph(lhs, "main");
  OptionalGraph rGraphOpt = mlir::hil::getGraph(rhs, "main");

  if (!lGraphOpt.has_value() || !rGraphOpt.has_value()) {
    std::cerr << ": One of models doesn't have main graph." << std::endl;
    return lGraphOpt.has_value() == rGraphOpt.has_value();
  }

  auto lGraph = lGraphOpt.value();
  auto rGraph = rGraphOpt.value();

  makeExprs(lGraph, ctx, nodes);
  makeExprs(rGraph, ctx, nodes);

  // create equations for graph inputs
  const NodeVector lInputs = getInputs(lGraph);
  const NodeVector rInputs = getInputs(rGraph);
  NodePairList sources;

  if (!match(lInputs, rInputs, sources)) {
    std::cerr << "Cannot match graphs inputs!" << std::endl;
    return false;
  }

  for (const auto &inPair : sources) {

    Node fIn = inPair.first;
    Node sIn = inPair.second;

    ChanVector fOuts = getOutputs(fIn);
    ChanVector sOuts = getOutputs(sIn);

    assert(fOuts.size() == sOuts.size());

    for (size_t i = 0; i < fOuts.size(); i++) {

      Chan fOut = fOuts[i];
      Chan sOut = sOuts[i];

      const Expr fFunc = toInFunc(fIn, fOut, ctx);
      const Expr sFunc = toInFunc(sIn, sOut, ctx);
      const Expr inEq = fFunc == sFunc;

      nodes.push_back(inEq);
    }
  }

  // create inequations for outputs
  const NodeVector lOuts = getSinks(lGraph);
  const NodeVector rOuts = getSinks(rGraph);
  NodePairList outMatch;

  if (!match(lOuts, rOuts, outMatch)) {

    std::cerr << "Cannot match graphs outputs!" << std::endl;
    return false;
  }

  for (const auto &outPair : outMatch) {

    Node fOut = outPair.first;
    Node sOut = outPair.second;

    // function names
    const std::string fName = fOut.nodeTypeName().str();
    const std::string sName = sOut.nodeTypeName().str();

    // output sorts
    const Sort fOutSort = getSort(fOut, ctx);
    const Sort sOutSort = getSort(sOut, ctx);

    // input sorts
    const SortVector fInSorts = getInSorts(fOut, ctx);
    const SortVector sInSorts = getInSorts(sOut, ctx);

    const FuncDecl fFunc = function(fName.c_str(), fInSorts, fOutSort);
    const ExprVector fArgs = getFuncArgs(fOut, ctx);

    const FuncDecl sFunc = function(sName.c_str(), sInSorts, sOutSort);
    const ExprVector sArgs = getFuncArgs(sOut, ctx);

    const Expr outExpr = fFunc(fArgs) != sFunc(sArgs);

    nodes.push_back(outExpr);
  }

  Solver solver(ctx);
  solver.add(nodes);

  std::cout << "SMT-LIBv2 formula:" << std::endl;
  std::cout << solver.to_smt2() << std::endl;

  CheckResult result = solver.check();
  switch (result) {
    case z3::sat:
      std::cout << "Models are NOT equivalent" << std::endl;
      std::cout << "Counterexample is:" << std::endl;
      std::cout << solver.get_model().to_string() << std::endl;
      return true;
    case z3::unsat:
      std::cout << "Models are equivalent" << std::endl;
      return false;
    case z3::unknown:
      std::cout << "Z3 solver returns \"unknown\"" << std::endl;
      return false;
      break;
  }
  std::cout << "Z3 returns unexpected result: " << result << std::endl;
  return false;
}

bool EqChecker::match(
    const NodeVector &lhs,
    const NodeVector &rhs,
    NodePairList &matched) const {

  size_t lSize = lhs.size();
  size_t rSize = rhs.size();

  if (lSize != rSize) {
    return false;
  }

  for (size_t i = 0; i < lSize; i++) {

    Node lNode = lhs[i];
    std::string lName = lNode.name().str();
    bool hasMatch = false;

    for (size_t j = 0; j < rSize; j++) {

      Node rNode = rhs[j];
      std::string rName = rNode.name().str();

      if (lName == rName) {

        matched.push_back(std::make_pair(lNode, rNode));
        hasMatch = true;
      }
    }
    if (!hasMatch) {

      std::cerr << ": No match for graph: " + lName << std::endl;
      return false;
    }
  }
  return true;
}

void EqChecker::makeExprs(Graph &graph, Context &ctx, ExprVector &nodes) const {

  // create equations for channels
  ChanVector gChannels = getChans(graph);

  for (auto &ch : gChannels) {

    Expr src = toConst(ch, ch.nodeFrom(), ctx);
    Expr dst = toConst(ch, ch.nodeTo(), ctx);
    Expr chConst = toConst(ch, ctx);

    const Expr srcExpr = src == chConst;
    nodes.push_back(srcExpr);
    const Expr dstExpr = chConst == dst;
    nodes.push_back(dstExpr);
  }

  // create equations for nodes
  auto &gNodes = getNodes(graph);

  for (auto &node_op : gNodes) {

    Node node = mlir::cast<Node>(node_op);

    if (isDelay(node)) {

      // treat delay node as in-to-out channel
      Chan input = getInputs(node)[0];
      Chan output = getOutputs(node)[0];

      const Expr in = toConst(input, input.nodeTo(), ctx);
      const Expr out = toConst(output, output.nodeFrom(), ctx);

      const Expr delayExpr = in == out;

      nodes.push_back(delayExpr);

    } else if (isKernel(node)) {

      std::vector<Chan> nodeOuts = getOutputs(node);

      // create equation for every output port of kernel node
      for (auto &nOut : nodeOuts) {

        // function name
        const Binding nOutBnd = nOut.nodeFrom();
        const Port port = nOutBnd.getPort();
        const std::string pName = port.getName();
        const std::string tName = node.nodeTypeName().str();
        const bool oneOut = nodeOuts.size() == 1;
        const std::string fName = oneOut ? tName : tName + "_" + pName;

        // input/output sorts for kernel function
        const SortVector sorts = getInSorts(node, ctx);
        const Sort fSort = getSort(port, ctx);

        // kernel function
        const FuncDecl kernel = function(fName.c_str(), sorts, fSort);
        const ExprVector kArgs = getFuncArgs(node, ctx);

        // create equation & store it
        const Expr kernEq = kernel(kArgs) == toConst(nOut, nOutBnd, ctx);
        nodes.push_back(kernEq);
      }
    } else if (isMerge(node)) {

      // merge node has the only output
      Chan nodeOut = getOutputs(node)[0];
      Binding outBnd = nodeOut.nodeFrom();

      const Expr outConst = toConst(nodeOut, outBnd, ctx);
      ExprVector mergeVec(ctx);

      ChanVector nodeInputs = getInputs(node);

      // create equation for every input of node
      for (auto &nodeInput : nodeInputs) {

        Binding inBnd = nodeInput.nodeTo();
        const Expr inConst = toConst(nodeInput, inBnd, ctx);

        mergeVec.push_back(outConst == inConst);
      }

      nodes.push_back(mk_or(mergeVec));

    } else if (isDup (node) || isSplit(node)) {

      // split node has the only input
      auto inputs = getInputs(node);
      assert(inputs.size() == 1);

      Chan nodeInput = inputs[0];
      Binding inBnd = nodeInput.nodeTo();

      const Expr inConst = toConst(nodeInput, inBnd, ctx);
      ExprVector splitVec(ctx);

      ChanVector nodeOutputs = getOutputs(node);

      // create equation for every output of node
      for (auto &nodeOut : nodeOutputs) {

        Binding outBnd = nodeOut.nodeFrom();
        const Expr outConst = toConst(nodeOut, outBnd, ctx);
        const Expr outEq = inConst == outConst;
        splitVec.push_back(outEq);
      }

      nodes.push_back(mk_and(splitVec));

    } else if (!isSink(node) && !isSource(node)) {

      // sink or source, do nothing
      const auto nodeType = node.nodeTypeName().str();
      std::cerr << "Unsupported node type: " + nodeType << std::endl;
    }
  }
}

Sort EqChecker::getSort(Node &node, Context &ctx) const {

  return ctx.uninterpreted_sort(node.nodeTypeName().str().c_str());
}

Sort EqChecker::getSort(Port port, Context &ctx) const {

  return ctx.uninterpreted_sort(port.getTypeName().c_str());
}

Expr EqChecker::toConst(Chan &ch, const Binding &bnd, Context &ctx) const {

  const auto port = bnd.getPort();
  const std::string name = port.getName();
  const Sort fInSort = getSort(port, ctx);
  const std::string modelName = getModelName(ch);
  const std::string nodeName = bnd.getNodeName().str();
  const std::string constName = modelName + "_" + nodeName + "_" + name;

  return ctx.constant(constName.c_str(), fInSort);
}

Expr EqChecker::toConst(Node &node, Context &ctx) const {

  const Sort fInSort = getSort(node, ctx);
  const std::string modelName = getModelName(node);
  const std::string constName = modelName + "_" + node.name().str();

  return ctx.constant(constName.c_str(), fInSort);
}

Expr EqChecker::toConst(Chan &ch, Context &ctx) const {

  const auto fromPort = ch.nodeFrom().getPort();
  const auto toPort = ch.nodeTo().getPort();
  assert(fromPort.getTypeName() == toPort.getTypeName());

  return ctx.constant(ch.varName().str().c_str(), getSort(fromPort, ctx));
}

Expr EqChecker::toInFunc(Node &node, Chan &ch, Context &ctx) const {

  const std::string modelName = getModelName(node);
  const std::string nodeName = node.name().str();
  const std::string chName = ch.nodeFrom().getPort().getName();
  const std::string funcName = modelName + "_" + nodeName + "_" + chName;
  const Sort fSort = getSort(ch.nodeFrom().getPort(), ctx);
  SortVector sorts(ctx);
  const FuncDecl func = function(funcName.c_str(), sorts, fSort);

  return func(ExprVector(ctx));
}

SortVector EqChecker::getInSorts(Node &node, Context &ctx) const {

  auto inputs = getInputs(node);
  SortVector sorts(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {

    Port targetPort = inputs[i].nodeTo().getPort();
    sorts.push_back(getSort(targetPort, ctx));
  }
  return sorts;
}

ExprVector EqChecker::getFuncArgs(Node &node, Context &ctx) const {

  ChanVector inputs = getInputs(node);
  ExprVector args(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {

    Binding bnd = inputs[i].nodeTo();
    args.push_back(toConst(inputs[i], bnd, ctx));
  }
  return args;
}

} // namespace eda::hls::eqchecker
