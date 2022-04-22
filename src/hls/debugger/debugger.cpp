//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
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
  std::cout << left.name().str() << std::endl;
  std::cout << right.name().str() << std::endl;

  mlir::hil::Graph *lGraph = mlir::hil::getGraph(left, "main");
  mlir::hil::Graph *rGraph = mlir::hil::getGraph(right, "main");

  createExprs(*lGraph, ctx, nodes);
  createExprs(*rGraph, ctx, nodes);

  // create equations for graph inputs
  const std::vector<mlir::hil::Node*> lInputs = getSources(*lGraph);
  const std::vector<mlir::hil::Node*> rInputs = getSources(*rGraph);
  std::list<std::pair<mlir::hil::Node*, mlir::hil::Node*>> sources;

  if (!match(lInputs, rInputs, sources)) {

    std::cout << "Cannot match graphs inputs!" << std::endl;
    return false;
  }

  for (const auto &inPair : sources) {

    mlir::hil::Node *fIn = inPair.first;
    mlir::hil::Node *sIn = inPair.second;

    std::vector<mlir::hil::Chan*> fOuts = getOutputs(*fIn);
    std::vector<mlir::hil::Chan*> sOuts = getOutputs(*sIn);

    assert(fOuts.size() == sOuts.size());

    for (size_t i = 0; i < fOuts.size(); i++) {

      mlir::hil::Chan *fOut = fOuts[i];
      mlir::hil::Chan *sOut = sOuts[i];

      const z3::expr fFunc = toInFunc(*fIn, *fOut, ctx);
      const z3::expr sFunc = toInFunc(*sIn, *sOut, ctx);
      const z3::expr inEq = fFunc == sFunc;

      nodes.push_back(inEq);
    }
  }

  // create inequations for outputs
  const std::vector<mlir::hil::Node*> lOuts = getSinks(*lGraph);
  const std::vector<mlir::hil::Node*> rOuts = getSinks(*rGraph);
  std::list<std::pair<mlir::hil::Node*, mlir::hil::Node*>> outMatch;

  if (!match(lOuts, rOuts, outMatch)) {

    std::cout << "Cannot match graphs outputs" << std::endl;
    return false;
  }

  for (const auto &outPair : outMatch) {

    mlir::hil::Node *fOut = outPair.first;
    mlir::hil::Node *sOut = outPair.second;

    // function names
    const std::string fName = getFuncName(*fOut);
    const std::string sName = getFuncName(*sOut);

    // output sorts
    const z3::sort fOutSort = getSort(*fOut, ctx);
    const z3::sort sOutSort = getSort(*sOut, ctx);

    // input sorts
    const z3::sort_vector fInSorts = getInSorts(*fOut, ctx);
    const z3::sort_vector sInSorts = getInSorts(*sOut, ctx);

    const z3::func_decl fFunc = function(fName.c_str(), fInSorts, fOutSort);
    const z3::expr_vector fArgs = getFuncArgs(*fOut, ctx);

    const z3::func_decl sFunc = function(sName.c_str(), sInSorts, sOutSort);
    const z3::expr_vector sArgs = getFuncArgs(*sOut, ctx);

    const z3::expr outExpr = fFunc(fArgs) != sFunc(sArgs);

    nodes.push_back(outExpr);
  }

  z3::solver solver(ctx);
  solver.add(nodes);

  // TODO: debug print
  //std::cout << "SMT-LIBv2 formula:" << std::endl;
  //std::cout << solver.to_smt2() << std::endl;

  z3::check_result result = solver.check();
  switch (result) {
    case z3::sat:
      std::cout << "Models are NOT equivalent" << std::endl;
      // TODO: debug print
      //std::cout << "Model is:" << std::endl;
      //std::cout << solver.get_model().to_string() << std::endl;
      return true;
    case z3::unsat:
      std::cout << "Models are equivalent" << std::endl;
      return false;
    case z3::unknown:
      std::cout << "Z3 solver says \"unknown\"" << std::endl;
      return false;
      break;
  }
  std::cout << "Z3 solver returns unexpected result: " << result << std::endl;
  return false;
}

bool EqChecker::match(const std::vector<Graph*> &left,
    const std::vector<Graph*> &right,
    std::pair<Graph*, Graph*> &matched) const {

  size_t lSize = left.size();
  size_t rSize = right.size();

  for (size_t i = 0; i < lSize; i++) {

    Graph *lGraph = left[i];

    if (!(lGraph->name() == "main")) {
      continue;
    }

    for (size_t j = 0; j < rSize; j++) {

      Graph *rGraph = right[j];

      if (!(rGraph->name() == "main")) {
        continue;
      }

      matched.first = lGraph;
      matched.second = rGraph;
      return true;
    }
  }
  return false;
}

bool EqChecker::match(const std::vector<mlir::hil::Node*> &left,
    const std::vector<mlir::hil::Node*> &right,
    std::list<std::pair<mlir::hil::Node*, mlir::hil::Node*>> &matched) const {
  size_t lSize = left.size();
  size_t rSize = right.size();

  if (lSize != rSize)
    return false;

  for (size_t i = 0; i < lSize; i++) {

    Node *lNode = left[i];
    bool hasMatch = false;

    for (size_t j = 0; j < rSize; j++) {
      Node *rNode = right[j];

      if (lNode->name() == rNode->name()) {

        matched.push_back(std::make_pair(lNode, rNode));
        hasMatch = true;
      }
    }
    if (!hasMatch) {

      std::cout << "No match for graphs " + lNode->name().str() << std::endl;
      return false;
    }
  }
  return true;
}

void EqChecker::createExprs(mlir::hil::Graph &graph,
    z3::context &ctx,
    z3::expr_vector &nodes) const {

  // create equations for channels
  std::vector<mlir::hil::Chan*> gChannels = getChans(graph);

  for (auto &channel : gChannels) {

    z3::expr src = toConst(*channel, channel->nodeFromAttrName(), ctx);
    z3::expr tgt = toConst(*channel, channel->nodeToAttrName(), ctx);

    const z3::expr chanExpr = src == tgt;
    nodes.push_back(chanExpr);
  }

  // create equations for nodes
  auto &graph_ops = graph.getBody()->getOperations();
  mlir::hil::Nodes nodes_op = find_elem_by_type<Nodes>(graph_ops).value();
  auto &gNodes = nodes_op.getBody()->getOperations();

  for (auto &node_op : gNodes) {

    mlir::hil::Node node = mlir::cast<Node>(node_op);

    if (isDelay(node)) {

      // treat delay node as in-to-out channel
      Chan *input = getInputs(node)[0];
      Chan *output = getOutputs(node)[0];

      const z3::expr in = toConst(*input, input->nodeToAttrName(), ctx);
      const z3::expr out = toConst(*output, output->nodeFromAttrName(), ctx);

      const z3::expr delayExpr = in == out;

      nodes.push_back(delayExpr);

    } else if (isKernel(node)) {

      const std::vector<Chan*> nodeOuts = getOutputs(node);

      // create equation for every output port of kernel node
      for (const auto &nodeOut : nodeOuts) {

        //const Binding srcBnd = nodeOut->source;

        // input/output sorts for kernel function
        const z3::sort_vector sorts = getInSorts(node, ctx);
        const z3::sort fSort = getSort(nodeOut->nodeFromAttrName(), ctx);

        // kernel function name
        //const char *kerName = node.nodeTypeName().str().c_str();

        // kernel function
        const z3::func_decl kernel = function(node.nodeTypeName().str().c_str(), sorts, fSort);
        const z3::expr_vector kernelArgs = getFuncArgs(node, ctx);

        // create equation
        const z3::expr kernelEq =
            kernel(kernelArgs) ==
                toConst(*nodeOut, nodeOut->nodeFromAttrName(), ctx);

        nodes.push_back(kernelEq);
      }
    } else if (isMerge(node)) {

      // merge node has the only output
      Chan *nodeOut = getOutputs(node)[0];

      const z3::expr outConst =
          toConst(*nodeOut, nodeOut->nodeFromAttrName(), ctx);
      z3::expr_vector mergeVec(ctx);

      const std::vector< Chan*> nodeInputs = getInputs(node);

      // create equation for every input of node
      for (const auto &nodeInput : nodeInputs) {

        const z3::expr inConst =
            toConst(*nodeInput, nodeInput->nodeToAttrName(), ctx);

        mergeVec.push_back(outConst == inConst);
      }

      nodes.push_back(mk_and(mergeVec));

    } else if (isSplit(node)) {

      // split node has the only input
      Chan *nodeInput = getInputs(node)[0];

      const z3::expr inConst =
          toConst(*nodeInput, nodeInput->nodeToAttrName(), ctx);
      z3::expr_vector splitVec(ctx);

      std::vector< Chan*> nodeOutputs = getOutputs(node);

      // create equation for every output of node
      for (auto &nodeOut : nodeOutputs) {

        const z3::expr outConst =
            toConst(*nodeOut, nodeOut->nodeFromAttrName(), ctx);
        const z3::expr outEq = inConst == outConst;
        splitVec.push_back(outEq);
      }

      nodes.push_back(mk_and(splitVec));
    } else {
      // sink or source, do nothing
      assert(isSink(node) || isSource(node));
    }
  }
}

std::vector<Node*> EqChecker::getSources(mlir::hil::Graph &graph) const {

  std::vector<Node*> result;
  std::vector<Node*> graphNodes = getNodes(graph);

  for (auto &node : graphNodes) {

    if (isSource(*node)) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<Node*> EqChecker::getSinks(mlir::hil::Graph &graph) const {

  std::vector<Node*> result;
  std::vector<Node*> graphNodes = getNodes(graph);

  for (auto &node : graphNodes) {

    if (isSink(*node)) {
      result.push_back(node);
    }
  }
  return result;
}

std::string EqChecker::getModelName(mlir::hil::Node &node) const {
  auto model =
      mlir::cast<Model>(*node->getParentOp()->getParentOp()->getParentOp());
  return model.name().str();
}

std::string EqChecker::getModelName(mlir::hil::Chan &ch) const {
  auto model =
      mlir::cast<Model>(*ch->getParentOp()->getParentOp());
  return model.name().str();
}

std::string EqChecker::getFuncName(mlir::hil::Node &node) const {
  return node.nodeTypeName().str();
}

z3::sort EqChecker::getSort(mlir::hil::Node &node, z3::context &ctx) const {
  return ctx.uninterpreted_sort(node.nodeTypeNameAttrName().str().c_str());
}

z3::sort EqChecker::getSort(mlir::StringAttr name, z3::context &ctx) const {
  return ctx.uninterpreted_sort(name.str().c_str());
}

z3::expr EqChecker::toConst(mlir::hil::Chan &ch, mlir::StringAttr name,
    z3::context &ctx) const {

  const z3::sort fInSort = getSort(name, ctx);
  const std::string modelName = getModelName(ch);
  const std::string nodeName = ch.varName().str();
  const std::string constName = modelName + "_" + nodeName + "_" + name.str();

  return ctx.constant(constName.c_str(), fInSort);
}

z3::expr EqChecker::toConst(mlir::hil::Node &node, z3::context &ctx) const {

  const z3::sort fInSort = getSort(node, ctx);
  const std::string modelName = getModelName(node);
  const std::string constName = modelName + "_" + node.name().str();

  return ctx.constant(constName.c_str(), fInSort);
}

z3::expr EqChecker::toInFunc(Node &node, Chan &ch, z3::context &ctx) const {

  const std::string modelName = getModelName(node);
  const std::string nodeName = node.name().str();
  const std::string chName = ch.varName().str();
  const std::string funcName = modelName + "_" + nodeName + "_" + chName;
  const z3::sort fSort = getSort(ch.nodeFromAttrName(), ctx);
  z3::sort_vector sorts(ctx);
  const z3::func_decl func = function(funcName.c_str(), sorts, fSort);
  const z3::expr_vector fArgs = z3::expr_vector(ctx);
  return func(fArgs);
}

z3::sort_vector EqChecker::getInSorts(mlir::hil::Node &node,
  z3::context &ctx) const {

  auto inputs = getInputs(node);
  z3::sort_vector sorts(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {
    mlir::StringAttr targetPortName = inputs[i]->nodeToAttrName();
    sorts.push_back(getSort(targetPortName, ctx));
  }
  return sorts;
}

z3::expr_vector EqChecker::getFuncArgs(Node &node, z3::context &ctx) const {

  std::vector<Chan*> inputs = getInputs(node);

  z3::expr_vector args(ctx);

  for (size_t i = 0; i < inputs.size(); i++) {
    mlir::StringAttr portName = inputs[i]->nodeToAttrName();
    args.push_back(toConst(*inputs[i], portName, ctx));
  }

  return args;
}

} // namespace eda::hls::debugger
