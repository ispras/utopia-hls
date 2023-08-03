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

bool EqChecker::equivalent(ModelOp &lhs, ModelOp &rhs) const {

  Context ctx;
  ExprVector nodes(ctx);

  OptionalGraphOp lGraphOpt = mlir::hil::getGraphOp(lhs, "main");
  OptionalGraphOp rGraphOpt = mlir::hil::getGraphOp(rhs, "main");

  if (!lGraphOpt.has_value() || !rGraphOpt.has_value()) {
    std::cerr << ": One of models doesn't have main graph." << std::endl;
    return lGraphOpt.has_value() == rGraphOpt.has_value();
  }

  auto lGraph = lGraphOpt.value();
  auto rGraph = rGraphOpt.value();

  makeExprs(lGraph, ctx, nodes);
  makeExprs(rGraph, ctx, nodes);

  // create equations for graph inputs
  const NodeOpVector lInputs = getSourcesAndConsts(lGraph);
  const NodeOpVector rInputs = getSourcesAndConsts(rGraph);
  NodeOpPairList sources;

  if (!match(lInputs, rInputs, sources)) {
    std::cerr << "Cannot match graphs inputs!" << std::endl;
    return false;
  }

  for (const auto &inPair : sources) {

    NodeOp fIn = inPair.first;
    NodeOp sIn = inPair.second;

    ChanOpVector fOuts = getOutputs(fIn);
    ChanOpVector sOuts = getOutputs(sIn);

    assert(fOuts.size() == sOuts.size());

    for (size_t i = 0; i < fOuts.size(); i++) {

      ChanOp fOut = fOuts[i];
      ChanOp sOut = sOuts[i];

      const Expr fFunc = toInFunc(fIn, fOut, ctx);
      const Expr sFunc = toInFunc(sIn, sOut, ctx);
      const Expr inEq = fFunc == sFunc;

      nodes.push_back(inEq);
    }
  }

  // create inequations for outputs
  const NodeOpVector lOuts = getSinks(lGraph);
  const NodeOpVector rOuts = getSinks(rGraph);
  NodeOpPairList outMatch;

  if (!match(lOuts, rOuts, outMatch)) {

    std::cerr << "Cannot match graphs outputs!" << std::endl;
    return false;
  }

  for (const auto &outPair : outMatch) {

    NodeOp fOut = outPair.first;
    NodeOp sOut = outPair.second;

    // function names
    const std::string fName = fOut.getNodeTypeName().str();
    const std::string sName = sOut.getNodeTypeName().str();

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
    const NodeOpVector &lhs,
    const NodeOpVector &rhs,
    NodeOpPairList &matched) const {

  size_t lSize = lhs.size();
  size_t rSize = rhs.size();

  if (lSize != rSize) {
    return false;
  }

  for (size_t i = 0; i < lSize; i++) {

    NodeOp lNode = lhs[i];
    std::string lName = lNode.getName().str();
    bool hasMatch = false;

    for (size_t j = 0; j < rSize; j++) {

      NodeOp rNode = rhs[j];
      std::string rName = rNode.getName().str();

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

void EqChecker::makeExprs(GraphOp &graphOp,
                          Context &ctx,
                          ExprVector &nodes) const {

  // create equations for channels
  ChanOpVector gChannels = getChans(graphOp);

  for (auto &chanOp : gChannels) {

    Expr src = toConst(chanOp, chanOp.getNodeFrom(), ctx);
    Expr dst = toConst(chanOp, chanOp.getNodeTo(), ctx);
    Expr chConst = toConst(chanOp, ctx);

    const Expr srcExpr = src == chConst;
    nodes.push_back(srcExpr);
    const Expr dstExpr = chConst == dst;
    nodes.push_back(dstExpr);
  }

  // create equations for nodes
  auto &gNodes = getNodes(graphOp);

  for (auto &node_op : gNodes) {

    NodeOp nodeOp = mlir::cast<NodeOp>(node_op);

    if (isDelay(nodeOp)) {

      // treat delay nodeOp as in-to-out channel
      ChanOp input = getInputs(nodeOp)[0];
      ChanOp output = getOutputs(nodeOp)[0];

      const Expr in = toConst(input, input.getNodeTo(), ctx);
      const Expr out = toConst(output, output.getNodeFrom(), ctx);

      const Expr delayExpr = in == out;

      nodes.push_back(delayExpr);

    } else if (isKernel(nodeOp)) {

      std::vector<ChanOp> nodeOuts = getOutputs(nodeOp);

      // create equation for every output port of kernel node
      for (auto &nOut : nodeOuts) {

        // function name
        const BindingAttr nOutBnd = nOut.getNodeFrom();
        const PortAttr port = nOutBnd.getPort();
        const std::string pName = port.getName();
        const std::string tName = nodeOp.getNodeTypeName().str();
        const bool oneOut = nodeOuts.size() == 1;
        const std::string fName = oneOut ? tName : tName + "_" + pName;

        // input/output sorts for kernel function
        const SortVector sorts = getInSorts(nodeOp, ctx);
        const Sort fSort = getSort(port, ctx);

        // kernel function
        const FuncDecl kernel = function(fName.c_str(), sorts, fSort);
        const ExprVector kArgs = getFuncArgs(nodeOp, ctx);

        // create equation & store it
        const Expr kernEq = kernel(kArgs) == toConst(nOut, nOutBnd, ctx);
        nodes.push_back(kernEq);
      }
    } else if (isMerge(nodeOp)) {

      // merge node has the only output
      ChanOp nodeOut = getOutputs(nodeOp)[0];
      BindingAttr outBnd = nodeOut.getNodeFrom();

      const Expr outConst = toConst(nodeOut, outBnd, ctx);
      ExprVector mergeVec(ctx);

      ChanOpVector nodeInputs = getInputs(nodeOp);

      // create equation for every input of node
      for (auto &nodeInput : nodeInputs) {

        BindingAttr inBnd = nodeInput.getNodeTo();
        const Expr inConst = toConst(nodeInput, inBnd, ctx);

        mergeVec.push_back(outConst == inConst);
      }

      nodes.push_back(mk_or(mergeVec));

    } else if (isDup (nodeOp) || isSplit(nodeOp)) {

      // split node has the only input
      auto inputs = getInputs(nodeOp);
      assert(inputs.size() == 1);

      ChanOp nodeInput = inputs[0];
      BindingAttr inBnd = nodeInput.getNodeTo();

      const Expr inConst = toConst(nodeInput, inBnd, ctx);
      ExprVector splitVec(ctx);

      ChanOpVector nodeOutputs = getOutputs(nodeOp);

      // create equation for every output of node
      for (auto &nodeOut : nodeOutputs) {

        BindingAttr outBnd = nodeOut.getNodeFrom();
        const Expr outConst = toConst(nodeOut, outBnd, ctx);
        const Expr outEq = inConst == outConst;
        splitVec.push_back(outEq);
      }

      nodes.push_back(mk_and(splitVec));

    } else if (!isSink(nodeOp) && !isSource(nodeOp)) {

      // sink or source, do nothing
      const auto nodeType = nodeOp.getNodeTypeName().str();
      std::cerr << "Unsupported node type: " + nodeType << std::endl;
    }
  }
}

Sort EqChecker::getSort(NodeOp &nodeOp, Context &ctx) const {

  return ctx.uninterpreted_sort(nodeOp.getNodeTypeName().str().c_str());
}

Sort EqChecker::getSort(PortAttr port, Context &ctx) const {

  return ctx.uninterpreted_sort(port.getTypeName().c_str());
}

Expr EqChecker::toConst(ChanOp &chanOp,
                        const BindingAttr &bnd,
                        Context &ctx) const {

  const auto port = bnd.getPort();
  const std::string name = port.getName();
  const Sort fInSort = getSort(port, ctx);
  const std::string modelName = getModelName(chanOp);
  const std::string nodeName = bnd.getNodeName().str();
  const std::string constName = modelName + "_" + nodeName + "_" + name;

  return ctx.constant(constName.c_str(), fInSort);
}

Expr EqChecker::toConst(NodeOp &nodeOp, Context &ctx) const {

  const Sort fInSort = getSort(nodeOp, ctx);
  const std::string modelName = getModelName(nodeOp);
  const std::string constName = modelName + "_" + nodeOp.getName().str();

  return ctx.constant(constName.c_str(), fInSort);
}

Expr EqChecker::toConst(ChanOp &chanOp, Context &ctx) const {

  const auto fromPort = chanOp.getNodeFrom().getPort();
  const auto toPort = chanOp.getNodeTo().getPort();
  assert(fromPort.getTypeName() == toPort.getTypeName());

  return ctx.constant(chanOp.getVarName().str().c_str(),
                      getSort(fromPort, ctx));
}

Expr EqChecker::toInFunc(NodeOp &nodeOp, ChanOp &chanOp, Context &ctx) const {

  const std::string modelOpName = getModelName(nodeOp);
  const std::string nodeOpName = nodeOp.getName().str();
  const std::string chanOpName = chanOp.getNodeFrom().getPort().getName();
  const std::string funcName = modelOpName + "_" + nodeOpName + "_"
                                           + chanOpName;
  const Sort fSort = getSort(chanOp.getNodeFrom().getPort(), ctx);
  SortVector sorts(ctx);
  const FuncDecl func = function(funcName.c_str(), sorts, fSort);

  return func(ExprVector(ctx));
}

SortVector EqChecker::getInSorts(NodeOp &nodeOp, Context &ctx) const {

  auto inputs = getInputs(nodeOp);
  SortVector sorts(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {

    PortAttr targetPort = inputs[i].getNodeTo().getPort();
    sorts.push_back(getSort(targetPort, ctx));
  }
  return sorts;
}

ExprVector EqChecker::getFuncArgs(NodeOp &nodeOp, Context &ctx) const {

  ChanOpVector inputs = getInputs(nodeOp);
  ExprVector args(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {

    BindingAttr bnd = inputs[i].getNodeTo();
    args.push_back(toConst(inputs[i], bnd, ctx));
  }
  return args;
}

} // namespace eda::hls::eqchecker