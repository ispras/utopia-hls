//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/debugger/debugger.h>

using namespace eda::hls::model;

namespace eda::hls::debugger {

std::unique_ptr<EqChecker> EqChecker::instance = nullptr;

bool EqChecker::equivalent(const Model &left, const Model &right) const {

  z3::context ctx;
  z3::expr_vector nodes(ctx);

  const Graph *lGraph = left.main();
  const Graph *rGraph = right.main();

  createExprs(*lGraph, ctx, nodes);
  createExprs(*rGraph, ctx, nodes);

  // create equations for graph inputs
  const std::vector<Node*> lInputs = getSources(*lGraph);
  const std::vector<Node*> rInputs = getSources(*rGraph);
  std::list<std::pair<Node*, Node*>> sources;

  if (!match(lInputs, rInputs, sources)) {

    std::cout << "Cannot match graphs inputs!" << std::endl;
    return false;
  }

  for (const auto &inPair : sources) {

    const Node *fIn = inPair.first;
    const Node *sIn = inPair.second;

    const std::vector<Chan*> fOuts = fIn->outputs;
    const std::vector<Chan*> sOuts = sIn->outputs;

    assert(fOuts.size() == sOuts.size());

    for (size_t i = 0; i < fOuts.size(); i++) {

      const Chan *fOut = fOuts[i];
      const Chan *sOut = sOuts[i];

      const z3::expr fFunc = toInFunc(*fIn, *fOut, ctx);
      const z3::expr sFunc = toInFunc(*sIn, *sOut, ctx);
      const z3::expr inEq = fFunc == sFunc;

      nodes.push_back(inEq);
    }
  }

  // create inequations for outputs
  const std::vector<Node*> lOuts = getSinks(*lGraph);
  const std::vector<Node*> rOuts = getSinks(*rGraph);
  std::list<std::pair<Node*, Node*>> outMatch;

  if (!match(lOuts, rOuts, outMatch)) {

    std::cout << "Cannot match graphs outputs" << std::endl;
    return false;
  }

  for (const auto &outPair : outMatch) {

    const Node *fOut = outPair.first;
    const Node *sOut = outPair.second;

    // function names
    const char *fName = fOut->type.name.c_str();
    const char *sName = sOut->type.name.c_str();

    // output sorts
    const z3::sort fOutSort = getSort(*fOut, ctx);
    const z3::sort sOutSort = getSort(*sOut, ctx);

    // input sorts
    const z3::sort_vector fInSorts = getInSorts(*fOut, ctx);
    const z3::sort_vector sInSorts = getInSorts(*sOut, ctx);

    const z3::func_decl fFunc = function(fName, fInSorts, fOutSort);
    const z3::expr_vector fArgs = getFuncArgs(*fOut, ctx);

    const z3::func_decl sFunc = function(sName, sInSorts, sOutSort);
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

    if (!lGraph->isMain()) {
      continue;
    }

    for (size_t j = 0; j < rSize; j++) {

      Graph *rGraph = right[j];

      if (!rGraph->isMain()) {
        continue;
      }

      matched.first = lGraph;
      matched.second = rGraph;
      return true;
    }
  }
  return false;
}

bool EqChecker::match(const std::vector<Node*> &left,
    const std::vector<Node*> &right,
    std::list<std::pair<Node*, Node*>> &matched) const {
  size_t lSize = left.size();
  size_t rSize = right.size();

  if (lSize != rSize)
    return false;

  for (size_t i = 0; i < lSize; i++) {

    Node *lNode = left[i];
    bool hasMatch = false;

    for (size_t j = 0; j < rSize; j++) {
      Node *rNode = right[j];

      if (lNode->name == rNode->name) {

        matched.push_back(std::make_pair(lNode, rNode));
        hasMatch = true;
      }
    }
    if (!hasMatch) {

      std::cout << "No match for graphs " + lNode->name << std::endl;
      return false;
    }
  }
  return true;
}

void EqChecker::createExprs(const Graph &graph,
    z3::context &ctx,
    z3::expr_vector &nodes) const {

  // create equations for channels
  const std::vector<Chan*> gChannels = graph.chans;

  for (const auto &channel : gChannels) {

    const z3::expr src = toConst(channel->source, ctx);
    const z3::expr tgt = toConst(channel->target, ctx);

    const z3::expr chanExpr = src == tgt;
    nodes.push_back(chanExpr);
  }

  // create equations for nodes
  const std::vector<Node*> gNodes = graph.nodes;

  for (const auto &node : gNodes) {

    if (node->isDelay()) {

      // treat delay node as in-to-out channel
      const Binding input = node->inputs[0]->target;
      const Binding output = node->outputs[0]->source;

      const z3::expr in = toConst(input, ctx);
      const z3::expr out = toConst(output, ctx);

      const z3::expr delayExpr = in == out;

      nodes.push_back(delayExpr);

    } else if (node->isKernel()) {

      const std::vector<Chan*> nodeOuts = node->outputs;

      // create equation for every output port of kernel node
      for (const auto &nodeOut : nodeOuts) {

        const Binding srcBnd = nodeOut->source;

        // input/output sorts for kernel function
        const z3::sort_vector sorts = getInSorts(*node, ctx);
        const z3::sort fSort = getSort(*srcBnd.port, ctx);

        // kernel function name
        const char *kerName = node->type.name.c_str();

        // kernel function
        const z3::func_decl kernel = function(kerName, sorts, fSort);
        const z3::expr_vector kernelArgs = getFuncArgs(*node, ctx);

        // create equation
        const z3::expr kernelEq = kernel(kernelArgs) == toConst(srcBnd, ctx);

        nodes.push_back(kernelEq);
      }
    } else if (node->isMerge()) {

      // merge has the only output
      const Chan *nodeOut = node->outputs[0];

      const z3::expr outConst = toConst(nodeOut->source, ctx);
      z3::expr_vector mergeVec(ctx);

      const std::vector< Chan*> nodeInputs = node->inputs;

      // create equation for every input of node
      for (const auto &nodeInput : nodeInputs) {

        const z3::expr inConst = toConst(nodeInput->target, ctx);

        mergeVec.push_back(outConst == inConst);
      }

      nodes.push_back(mk_and(mergeVec));

    } else if (node->isSplit()) {

      // split has the only input
      const Chan *nodeInput = node->inputs[0];

      const z3::expr inConst = toConst(nodeInput->target, ctx);
      z3::expr_vector splitVec(ctx);

      std::vector< Chan*> nodeOutputs = node->outputs;

      // create equation for every output of node
      for (const auto &nodeOut : nodeOutputs) {

        const z3::expr outConst = toConst(nodeOut->source, ctx);
        const z3::expr outEq = inConst == outConst;
        splitVec.push_back(outEq);
      }

      nodes.push_back(mk_and(splitVec));
    } else {
      // sink or source, do nothing
      assert(node->isSink() || node->isSource());
    }
  }
}

std::vector<Node*> EqChecker::getSources(const Graph &graph) const {

  std::vector<Node*> result;
  std::vector<Node*> graphNodes = graph.nodes;

  for (const auto &node : graphNodes) {

    if (node->isSource()) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<Node*> EqChecker::getSinks(const Graph &graph) const {

  std::vector<Node*> result;
  std::vector<Node*> graphNodes = graph.nodes;

  for (const auto &node : graphNodes) {

    if (node->isSink()) {
      result.push_back(node);
    }
  }
  return result;
}

std::string EqChecker::getModelName(const Node &node) const {
  return node.graph.model.name;
}

z3::sort EqChecker::getSort(const Node &node, z3::context &ctx) const {
  return ctx.uninterpreted_sort(node.type.name.c_str());
}

z3::sort EqChecker::getSort(const Port &port, z3::context &ctx) const {
  return ctx.uninterpreted_sort(port.type.name.c_str());
}

z3::expr EqChecker::toConst(const Binding &bnd, z3::context &ctx) const {

  const z3::sort fInSort = getSort(*bnd.port, ctx);
  const std::string modelName = getModelName(*bnd.node);
  const std::string nodeName = bnd.node->name;
  const std::string portName = bnd.port->name;
  const std::string constName = modelName + "_" + nodeName + "_" + portName;

  return ctx.constant(constName.c_str(), fInSort);
}

z3::expr EqChecker::toConst(const Node &node, z3::context &ctx) const {

  const z3::sort fInSort = getSort(node, ctx);
  const std::string modelName = getModelName(node);
  const std::string constName = modelName + "_" + node.name;

  return ctx.constant(constName.c_str(), fInSort);
}

z3::expr EqChecker::toInFunc(const Node &node, const Chan &ch,
    z3::context &ctx) const {

  const std::string modelName = getModelName(node);
  const std::string nodeName = node.name;
  const std::string funcName = modelName + "_" + nodeName + "_" + ch.name;
  const z3::sort fSort = getSort(*ch.source.port, ctx);
  z3::sort_vector sorts(ctx);
  const z3::func_decl func = function(funcName.c_str(), sorts, fSort);
  const z3::expr_vector fArgs = z3::expr_vector(ctx);
  return func(fArgs);
}

z3::sort_vector EqChecker::getInSorts(const Node &node,
  z3::context &ctx) const {

  const unsigned arity = node.inputs.size();
  z3::sort_vector sorts(ctx);

  for (size_t i = 0; i < arity; i++) {
    sorts.push_back(getSort(*node.inputs[i]->target.port, ctx));
  }
  return sorts;
}

z3::expr_vector EqChecker::getFuncArgs(const Node &node,
    z3::context &ctx) const {

  const std::vector<Chan*> inputs = node.inputs;
  const unsigned arity = inputs.size();

  z3::expr_vector args(ctx);

  for (size_t i = 0; i < arity; i++) {
    args.push_back(toConst(inputs[i]->target, ctx));
  }

  return args;
}

} // namespace eda::hls::debugger
