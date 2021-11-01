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

  std::unique_ptr<Verifier> Verifier::instance = nullptr;

  bool Verifier::equivalent(const Model &left, const Model &right) const {

    std::cout << "Graph matching.." << "\n";
    std::pair<Graph*, Graph*> pair;

    // If graphs are not completely matched
    if (!match(left.graphs, right.graphs, pair)) {
      std::cout << "Cannot match graphs!" << "\n";
      return false;
    }

    z3::context ctx;
    z3::expr_vector nodes(ctx);

    std::cout << "Check graph equivalence..." << "\n";

    const Graph *fGraph = pair.first;
    const Graph *sGraph = pair.second;

    std::cout << "generate expr for graph: " << fGraph->name << "\n";
    createExprs(*fGraph, ctx, nodes);

    std::cout << "generate expr for graph: " << sGraph->name << "\n";
    createExprs(*sGraph, ctx, nodes);

    std::cout << "add input bindings to nodes" << "\n";

    const std::vector<Node*> fInputs = getSources(*fGraph);
    const std::vector<Node*> sInputs = getSources(*sGraph);
    std::list<std::pair<Node*, Node*>> sources;

    if (match(fInputs, sInputs, sources)) {

      for (const auto &inPair : sources) {

        const Node *fIn = inPair.first;
        const Node *sIn = inPair.second;

        const std::vector<Chan*> fOuts = fIn->outputs;
        const std::vector<Chan*> sOuts = sIn->outputs;

        assert(fOuts.size() == sOuts.size());

        for (size_t i = 0; i < fOuts.size(); i++) {

          const Chan *fOut = fOuts[i];
          const Chan *sOut = sOuts[i];

          const z3::expr fFunc = toFunc(*fIn, *fOut, ctx);
          const z3::expr sFunc = toFunc(*sIn, *sOut, ctx);
          const z3::expr inEq = fFunc == sFunc;

          nodes.push_back(inEq);
        }
      }
    } else {
      std::cout << "Cannot match graphs inputs" << "\n";
    }

    std::cout << "add output bindings to nodes" << "\n";

    const std::vector<Node*> fOutputs = getSinks(*fGraph);
    const std::vector<Node*> sOutputs = getSinks(*sGraph);
    std::list<std::pair<Node*, Node*>> outMatch;

    if (match(fOutputs, sOutputs, outMatch)) {

      for (const auto &outPair : outMatch) {

        const Node *fOut = outPair.first;
        const Node *sOut = outPair.second;

        const std::string fModelName = fOut->graph.model.name;
        const std::string sModelName = sOut->graph.model.name;
        const std::string fName = fModelName + "_" + fOut->name;
        const std::string sName = sModelName + "_" + sOut->name;

        const z3::sort fSort = getSort(*fOut, ctx);
        const z3::sort sSort = getSort(*sOut, ctx);

        const z3::sort_vector fInSorts = getInSorts(*fOut, ctx);
        const z3::sort_vector sInSorts = getInSorts(*sOut, ctx);

        const z3::func_decl fFunc = function(fName.c_str(), fInSorts, fSort);
        const z3::expr_vector fArgs = getFuncArgs(*fOut, ctx);
        const z3::func_decl sFunc = function(sName.c_str(), sInSorts, sSort);
        const z3::expr_vector sArgs = getFuncArgs(*sOut, ctx);

        const z3::expr outExpr = fFunc(fArgs) != sFunc(sArgs);

        nodes.push_back(outExpr);
      }
    } else {
      std::cout << "Cannot match graphs outputs" << "\n";
    }

    z3::solver solver(ctx);
    solver.add(nodes);

    std::cout << "SMT-LIBv2 formula:" << "\n";
    std::cout << solver.to_smt2() << "\n";

    switch (solver.check()) {
      case z3::sat:
        std::cout << "Models are NOT equivalent" << "\n";
        // TODO: debug print
        //std::cout << "Model is:" << "\n";
        //std::cout << solver.get_model().to_string() << "\n";
        return true;
      case z3::unsat:
        std::cout << "Models are equivalent" << "\n";
        return false;
      case z3::unknown:
        std::cout << "Z3 solver says \"unknown\"" << "\n";
        return false;
        break;
    }
    std::cout << "Z3 solver returns unexpected result" << "\n";
    return false;
  }

  bool Verifier::match(const std::vector<Graph*> &left,
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

  bool Verifier::match(const std::vector<Node*> &left,
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
        std::cout << "No match for graphs " + lNode->name << "\n";
        return false;
      }
    }
    return true;
  }

  void Verifier::createExprs(const Graph &graph,
    z3::context &ctx,
    z3::expr_vector &nodes) const {

    std::cout << "Create equations for channels: " + graph.name << "\n";

    const std::vector<Chan*> gChannels = graph.chans;

    for (const auto &channel : gChannels) {

      const z3::expr src = toConst(channel->source, ctx);
      const z3::expr tgt = toConst(channel->target, ctx);

      const z3::expr chanExpr = src == tgt;
      nodes.push_back(chanExpr);
    }

    std::cout << "Create equations for nodes: " + graph.name << "\n";
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

        const std::string funcName = node->name;
        const std::vector<Chan*> nodeOuts = node->outputs;

        for (const auto &nodeOut : nodeOuts) {

          const Port *outPort = nodeOut->source.port;
          const std::string funcIdxName = nodeOut->name;
          const std::string modelName = nodeOut->graph.model.name;
          const std::string modelFuncPrefix = modelName + "_" + funcName;
          const z3::sort_vector sorts = getInSorts(*node, ctx);
          const std::string kerName = modelFuncPrefix + "_" + funcIdxName;
          const char *sortName =outPort->type.c_str();
          const z3::sort fSort = ctx.uninterpreted_sort(sortName);
          const z3::func_decl kernel = function(kerName.c_str(), sorts, fSort);
          const z3::expr_vector kernelArgs = getFuncArgs(*node, ctx);
          const std::string constName = modelFuncPrefix + "_" + outPort->type;
          const z3::expr nodeOutConst = ctx.constant(constName.c_str(), fSort);
          const z3::expr kernelEq = kernel(kernelArgs) == nodeOutConst;

          nodes.push_back(kernelEq);
        }
      } else if (node->isMerge()) {

        // merge has the only output
        const Chan *nodeOut = node->outputs[0];
        const char *outSortName = nodeOut->source.node->type.name.c_str();
        const z3::sort outSort = ctx.uninterpreted_sort(outSortName);

        const z3::expr outConst = toConst(nodeOut->source, ctx);

        z3::expr_vector mergeVec(ctx);
        const std::vector< Chan*> nodeInputs = node->inputs;

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

  std::vector<Node*> Verifier::getSources(const Graph &graph) const {

    std::vector<Node*> result;
    std::vector<Node*> graphNodes = graph.nodes;

    for (const auto &node : graphNodes) {

      if (node->isSource()) {
        result.push_back(node);
      }
    }
    return result;
  }

  std::vector<Node*> Verifier::getSinks(const Graph &graph) const {

    std::vector<Node*> result;
    std::vector<Node*> graphNodes = graph.nodes;

    for (const auto &node : graphNodes) {

      if (node->isSink()) {
        result.push_back(node);
      }
    }
    return result;
  }

  z3::sort Verifier::getSort(const Node &node, z3::context &ctx) const {
    return ctx.uninterpreted_sort(node.type.name.c_str());
  }

  z3::expr Verifier::toConst(const Binding &bnd, z3::context &ctx) const {

    const char *typeName = bnd.port->type.c_str();
    const z3::sort fInSort = ctx.uninterpreted_sort(typeName);
    const std::string modelName = bnd.node->graph.model.name;
    const std::string nodeName = bnd.node->name;
    const std::string constName =
        modelName + "_" + nodeName + "_" + bnd.port->name;

    return ctx.constant(constName.c_str(), fInSort);
  }

  z3::expr Verifier::toConst(const Node &node, z3::context &ctx) const {

    const NodeType &fType = node.type;
    const std::string typeName = fType.name;
    const z3::sort fInSort = ctx.uninterpreted_sort(typeName.c_str());
    const std::string modelName = node.graph.model.name;
    const std::string constName = modelName + "_" + node.name;

    return ctx.constant(constName.c_str(), fInSort);
  }

  z3::expr Verifier::toFunc(const Node &node, const Chan &ch,
      z3::context &ctx) const {

    const Binding src = ch.source;
    const Port *fPort = src.port;
    const std::string outIdx = ch.name;
    const std::string modelName = node.graph.model.name;
    const std::string nodeName = node.name;
    const std::string funcName = modelName + "_" + nodeName + "_" + outIdx;
    const char *sortName = fPort->type.c_str();
    const z3::sort fSort = ctx.uninterpreted_sort(sortName);
    z3::sort_vector sorts(ctx);
    const z3::func_decl func = function(funcName.c_str(), sorts, fSort);
    const z3::expr_vector fArgs = z3::expr_vector(ctx);
    return func(fArgs);
  }

  z3::sort_vector Verifier::getInSorts(const Node &node,
    z3::context &ctx) const {

    const unsigned arity = node.inputs.size();
    z3::sort_vector sorts(ctx);

    for (size_t i = 0; i < arity; i++) {

      const char *sortName = node.inputs[i]->target.port->type.c_str();
      sorts.push_back(ctx.uninterpreted_sort(sortName));
    }
    return sorts;
  }

  z3::expr_vector Verifier::getFuncArgs(const Node &node,
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
