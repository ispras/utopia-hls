//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "HIL/Ops.h"
#include "HIL/Utils.h"
#include "hls/debugger/debugger.h"

#include <iostream>

namespace eda::hls::debugger {

std::unique_ptr<EqChecker> EqChecker::instance = nullptr;

bool EqChecker::equivalent(mlir::hil::Model &left,
    mlir::hil::Model &right) const {

  z3::context ctx;
  z3::expr_vector nodes(ctx);

  std::optional<mlir::hil::Graph> lGraph = mlir::hil::getGraph(left, "main");
  std::optional<mlir::hil::Graph> rGraph = mlir::hil::getGraph(right, "main");

  if (!lGraph.has_value() || !rGraph.has_value()) {
    std::cerr << ": One of models doesn't have main graph." << std::endl;
    return lGraph.has_value() == rGraph.has_value();
  }

  createExprs(lGraph.value(), ctx, nodes);
  createExprs(rGraph.value(), ctx, nodes);

  // create equations for graph inputs
  const std::vector<mlir::hil::Node> lInputs = getSources(*lGraph);
  const std::vector<mlir::hil::Node> rInputs = getSources(*rGraph);
  std::list<std::pair<mlir::hil::Node, mlir::hil::Node>> sources;

  if (!match(lInputs, rInputs, sources)) {
    std::cerr << "Cannot match graphs inputs!" << std::endl;
    return false;
  }

  for (const auto &inPair : sources) {

    mlir::hil::Node fIn = inPair.first;
    mlir::hil::Node sIn = inPair.second;

    std::vector<mlir::hil::Chan> fOuts = getOutputs(fIn);
    std::vector<mlir::hil::Chan> sOuts = getOutputs(sIn);

    assert(fOuts.size() == sOuts.size());

    for (size_t i = 0; i < fOuts.size(); i++) {

      mlir::hil::Chan fOut = fOuts[i];
      mlir::hil::Chan sOut = sOuts[i];

      const z3::expr fFunc = toInFunc(fIn, fOut, ctx);
      const z3::expr sFunc = toInFunc(sIn, sOut, ctx);
      const z3::expr inEq = fFunc == sFunc;

      nodes.push_back(inEq);
    }
  }

  // create inequations for outputs
  const std::vector<mlir::hil::Node> lOuts = getSinks(lGraph.value());
  const std::vector<mlir::hil::Node> rOuts = getSinks(rGraph.value());
  std::list<std::pair<mlir::hil::Node, mlir::hil::Node>> outMatch;

  if (!match(lOuts, rOuts, outMatch)) {

    std::cerr << "Cannot match graphs outputs!" << std::endl;
    return false;
  }

  for (const auto &outPair : outMatch) {

    mlir::hil::Node fOut = outPair.first;
    mlir::hil::Node sOut = outPair.second;

    // function names
    const std::string fName = getFuncName(fOut);
    const std::string sName = getFuncName(sOut);

    // output sorts
    const z3::sort fOutSort = getSort(fOut, ctx);
    const z3::sort sOutSort = getSort(sOut, ctx);

    // input sorts
    const z3::sort_vector fInSorts = getInSorts(fOut, ctx);
    const z3::sort_vector sInSorts = getInSorts(sOut, ctx);

    const z3::func_decl fFunc = function(fName.c_str(), fInSorts, fOutSort);
    const z3::expr_vector fArgs = getFuncArgs(fOut, ctx);

    const z3::func_decl sFunc = function(sName.c_str(), sInSorts, sOutSort);
    const z3::expr_vector sArgs = getFuncArgs(sOut, ctx);

    const z3::expr outExpr = fFunc(fArgs) != sFunc(sArgs);

    nodes.push_back(outExpr);
  }

  z3::solver solver(ctx);
  solver.add(nodes);

  // TODO: debug print
  std::cout << "SMT-LIBv2 formula:" << std::endl;
  std::cout << solver.to_smt2() << std::endl;

  z3::check_result result = solver.check();
  switch (result) {
    case z3::sat:
      std::cout << "Models are NOT equivalent" << std::endl;
      std::cout << "Model is:" << std::endl;
      std::cout << solver.get_model().to_string() << std::endl;
      return true;
    case z3::unsat:
      std::cout << "Models are equivalent" << std::endl;
      return false;
    case z3::unknown:
      std::cout << ": Z3 solver returns \"unknown\"" << std::endl;
      return false;
      break;
  }
  std::cout << ": Z3 returns unexpected result: " << result << std::endl;
  return false;
}

bool EqChecker::match(const std::vector<mlir::hil::Node> &left,
    const std::vector<mlir::hil::Node> &right,
    std::list<std::pair<mlir::hil::Node, mlir::hil::Node>> &matched) const {
  size_t lSize = left.size();
  size_t rSize = right.size();

  if (lSize != rSize)
    return false;

  for (size_t i = 0; i < lSize; i++) {

    mlir::hil::Node lNode = left[i];
    std::string lName = lNode.name().str();
    bool hasMatch = false;

    for (size_t j = 0; j < rSize; j++) {
      mlir::hil::Node rNode = right[j];
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

void EqChecker::createExprs(mlir::hil::Graph &graph,
    z3::context &ctx,
    z3::expr_vector &nodes) const {

  // create equations for channels
  std::vector<mlir::hil::Chan> gChannels = getChans(graph);

  for (auto &ch : gChannels) {

    z3::expr src = toConst(ch, ch.nodeFrom(), ctx);
    z3::expr dst = toConst(ch, ch.nodeTo(), ctx);
    z3::expr chConst = toConst(ch, ctx);

    const z3::expr srcExpr = src == chConst;
    nodes.push_back(srcExpr);
    const z3::expr dstExpr = chConst == dst;
    nodes.push_back(dstExpr);
  }

  // create equations for nodes
  auto &gNodes = getNodes(graph);

  for (auto &node_op : gNodes) {

    mlir::hil::Node node = mlir::cast<Node>(node_op);

    if (isDelay(node)) {

      // treat delay node as in-to-out channel
      mlir::hil::Chan input = getInputs(node)[0];
      mlir::hil::Chan output = getOutputs(node)[0];

      const z3::expr in = toConst(input, input.nodeTo(), ctx);
      const z3::expr out = toConst(output, output.nodeFrom(), ctx);

      const z3::expr delayExpr = in == out;

      nodes.push_back(delayExpr);

    } else if (isKernel(node)) {

      std::vector<Chan> nodeOuts = getOutputs(node);

      // create equation for every output port of kernel node
      for (auto &nOut : nodeOuts) {

        // function name
        const mlir::hil::BindingAttr nOutBnd = nOut.nodeFrom();
        const mlir::hil::PortAttr port = nOutBnd.getPort();
        const std::string pName = port.getName();
        const std::string tName = node.nodeTypeName().str();
        const bool oneOut = nodeOuts.size() == 1;
        const std::string fName = oneOut ? tName : tName + "_" + pName;

        // input/output sorts for kernel function
        const z3::sort_vector sorts = getInSorts(node, ctx);
        const z3::sort fSort = getSort(port, ctx);

        // kernel function
        const z3::func_decl kernel = function(fName.c_str(), sorts, fSort);
        const z3::expr_vector kArgs = getFuncArgs(node, ctx);

        // create equation & store it
        const z3::expr kernEq = kernel(kArgs) == toConst(nOut, nOutBnd, ctx);
        nodes.push_back(kernEq);
      }
    } else if (isMerge(node)) {

      // merge node has the only output
      mlir::hil::Chan nodeOut = getOutputs(node)[0];
      mlir::hil::BindingAttr outBnd = nodeOut.nodeFrom();

      const z3::expr outConst = toConst(nodeOut, outBnd, ctx);
      z3::expr_vector mergeVec(ctx);

      std::vector<mlir::hil::Chan> nodeInputs = getInputs(node);

      // create equation for every input of node
      for (auto &nodeInput : nodeInputs) {

        mlir::hil::BindingAttr inBnd = nodeInput.nodeTo();
        const z3::expr inConst = toConst(nodeInput, inBnd, ctx);

        mergeVec.push_back(outConst == inConst);
      }

      nodes.push_back(mk_or(mergeVec));

    } else if (isSplit(node)) {

      // split node has the only input
      auto inputs = getInputs(node);
      assert (inputs.size() == 1);
      mlir::hil::Chan nodeInput = inputs[0];
      mlir::hil::BindingAttr inBnd = nodeInput.nodeTo();

      const z3::expr inConst = toConst(nodeInput, inBnd, ctx);
      z3::expr_vector splitVec(ctx);

      std::vector<mlir::hil::Chan> nodeOutputs = getOutputs(node);

      // create equation for every output of node
      for (auto &nodeOut : nodeOutputs) {

        mlir::hil::BindingAttr outBnd = nodeOut.nodeFrom();
        const z3::expr outConst = toConst(nodeOut, outBnd, ctx);
        const z3::expr outEq = inConst == outConst;
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

std::string EqChecker::getModelName(mlir::hil::Node &node) const {
  auto model =
      mlir::cast<Model>(*node->getParentOp()->getParentOp()->getParentOp());
  return model.name().str();
}

std::string EqChecker::getModelName(mlir::hil::Chan &ch) const {
  auto model =
      mlir::cast<Model>(*ch->getParentOp()->getParentOp()->getParentOp());
  return model.name().str();
}

std::string EqChecker::getFuncName(mlir::hil::Node &node) const {
  return node.nodeTypeName().str();
}

z3::sort EqChecker::getSort(mlir::hil::Node &node, z3::context &ctx) const {
  return ctx.uninterpreted_sort(node.nodeTypeName().str().c_str());
}

z3::sort EqChecker::getSort(mlir::hil::PortAttr port, z3::context &ctx) const {
  return ctx.uninterpreted_sort(port.getTypeName().c_str());
}

z3::expr EqChecker::toConst(mlir::hil::Chan &ch,
    const mlir::hil::BindingAttr &bnd,
    z3::context &ctx) const {

  const auto port = bnd.getPort();
  std::string name = port.getName();
  const z3::sort fInSort = getSort(port, ctx);
  const std::string modelName = getModelName(ch);
  const std::string nodeName = bnd.getNodeName().str();
  const std::string constName = modelName + "_" + nodeName + "_" + name;

  return ctx.constant(constName.c_str(), fInSort);
}

z3::expr EqChecker::toConst(mlir::hil::Node &node, z3::context &ctx) const {

  const z3::sort fInSort = getSort(node, ctx);
  const std::string modelName = getModelName(node);
  const std::string constName = modelName + "_" + node.name().str();

  return ctx.constant(constName.c_str(), fInSort);
}

z3::expr EqChecker::toConst(mlir::hil::Chan &ch, z3::context &ctx) const {

  const auto fromPort = ch.nodeFrom().getPort();
  const auto toPort = ch.nodeTo().getPort();
  assert(fromPort.getTypeName() == toPort.getTypeName());

  return ctx.constant(ch.varName().str().c_str(), getSort(fromPort, ctx));
}

z3::expr EqChecker::toInFunc(mlir::hil::Node &node, mlir::hil::Chan &ch,
    z3::context &ctx) const {

  const std::string modelName = getModelName(node);
  const std::string nodeName = node.name().str();
  const std::string chName = ch.nodeFrom().getPort().getName();
  const std::string funcName = modelName + "_" + nodeName + "_" + chName;
  const z3::sort fSort = getSort(ch.nodeFrom().getPort(), ctx);
  z3::sort_vector sorts(ctx);
  const z3::func_decl func = function(funcName.c_str(), sorts, fSort);

  return func(z3::expr_vector(ctx));
}

z3::sort_vector EqChecker::getInSorts(mlir::hil::Node &node,
    z3::context &ctx) const {

  auto inputs = getInputs(node);
  z3::sort_vector sorts(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {
    mlir::hil::PortAttr targetPort = inputs[i].nodeTo().getPort();
    sorts.push_back(getSort(targetPort, ctx));
  }
  return sorts;
}

z3::expr_vector EqChecker::getFuncArgs(mlir::hil::Node &node,
    z3::context &ctx) const {

  std::vector<mlir::hil::Chan> inputs = getInputs(node);

  z3::expr_vector args(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {
    mlir::hil::BindingAttr bnd = inputs[i].nodeTo();
    args.push_back(toConst(inputs[i], bnd, ctx));
  }
  return args;
}

} // namespace eda::hls::debugger
