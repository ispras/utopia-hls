//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/debugger/debugger.h>

using namespace eda::hls::model;
using namespace z3;

namespace eda::hls::debugger {

  std::unique_ptr<Verifier> Verifier::instance = nullptr;

  bool Verifier::equivalent(Model &left, Model &right) const {

    std::cout << "Graph matching.." << "\n";
    std::list<std::pair<Graph*, Graph*>> matchedGraphs;

    // If graphs are not completely matched
    if (!match(left.graphs, right.graphs, matchedGraphs)) {
      std::cout << "Cannot match graphs!" << "\n";
      return false;
    }

    context ctx;
    expr_vector nodes(ctx);

    std::cout << "Check graph equivalence..." << "\n";
    for (const auto &pair : matchedGraphs) {

      Graph *fGraph = pair.first;
      Graph *sGraph = pair.second;

      std::cout << "generate expr for graph: " << fGraph->name << "\n";
      to_expr(fGraph, ctx, nodes);
      std::cout << "generate expr for graph: " << sGraph->name << "\n";
      to_expr(sGraph, ctx, nodes);

      std::cout << "add input bindings to nodes" << "\n";
      std::vector<Node*> fInputs = getSources(fGraph);
      std::vector<Node*> sInputs = getSources(sGraph);
      std::list<std::pair<Node*, Node*>> sources;

      if (match(fInputs, sInputs, sources)) {

        for (const auto &inPair : sources) {

          Node *fIn = inPair.first;
          Node *sIn = inPair.second;
          std::string fModelName = fIn->graph.model.name;
          std::string sModelName = sIn->graph.model.name;
          std::vector<Chan*> fOuts = fIn->outputs;
          std::vector<Chan*> sOuts = sIn->outputs;
          assert(fOuts.size() == sOuts.size());

          for (size_t i = 0; i < fOuts.size(); i++) {

            Chan *fOut = fOuts[i];
            expr fFunc = toFunc(fIn, fOut, ctx);
            Chan *sOut = sOuts[i];
            expr sFunc = toFunc(sIn, sOut, ctx);
            expr inEq = fFunc == sFunc;
            nodes.push_back(inEq);
          }
        }
      } else {
        std::cout << "Cannot match graphs inputs" << "\n";
      }

      std::cout << "add output bindings to nodes" << "\n";
      std::vector<Node*> fOutputs = getSinks(fGraph);
      std::vector<Node*> sOutputs = getSinks(sGraph);
      std::list<std::pair<Node*, Node*>> outMatch;

      if (match(fOutputs, sOutputs, outMatch)) {

        for (const auto &outPair : outMatch) {

          Node *fOut = outPair.first;
          Node *sOut = outPair.second;

          std::string fModelName = fOut->graph.model.name;
          std::string sModelName = sOut->graph.model.name;
          std::string fName = fOut->name + "_" + fModelName;
          std::string sName = sOut->name + "_" + sModelName;

          sort fSort = getSort(fOut, ctx);
          sort sSort = getSort(sOut, ctx);

          sort_vector fInSorts = getInSorts(fOut, ctx);
          sort_vector sInSorts = getInSorts(sOut, ctx);

          func_decl fFunc = function(fName.c_str(), fInSorts, fSort);
          expr_vector fArgs = getArgs(fOut, ctx);
          func_decl sFunc = function(sName.c_str(), sInSorts, sSort);
          expr_vector sArgs = getArgs(sOut, ctx);

          expr outExpr = fFunc(fArgs) != sFunc(sArgs);

          nodes.push_back(outExpr);
        }
      } else {
        std::cout << "Cannot match graphs outputs" << "\n";
      }
    }

    solver solver(ctx);
    solver.add(nodes);

    std::cout << "SMT-LIBv2 formula:" << "\n";
    std::cout << solver.to_smt2() << "\n";

    switch (solver.check()) {
      case sat:
        std::cout << "Models are NOT equivalent" << "\n";
        std::cout << "Model is:" << "\n";
        std::cout << solver.get_model().to_string() << "\n";
        return true;
      case unsat:
        std::cout << "Models are equivalent" << "\n";
        return false;
      case unknown:
        std::cout << "Z3 solver says \"unknown\"" << "\n";
        return false;
        break;
    }
    std::cout << "Z3 solver returns unexpected result" << "\n";
    return false;
  }

  bool Verifier::match(std::vector<Graph*> left,
      std::vector<Graph*> right,
      std::list<std::pair<Graph*, Graph*>> &matches) const {

    size_t lSize = left.size();
    size_t rSize = right.size();

    if (lSize != rSize) {
      std::cout << "Graph collections are of different size!" << "\n";
      return false;
    }

    for (size_t i = 0; i < lSize; i++) {
      Graph *lGraph = left[i];
      bool hasMatch = false;

      for (size_t j = 0; j < rSize; j++) {
        Graph *rGraph = right[j];

        if (lGraph->name == rGraph->name) {
          matches.push_back(std::make_pair(lGraph, rGraph));
          hasMatch = true;
        }
      }
      if (!hasMatch) {
        std::cout << "No match for graphs " + lGraph->name << "\n";
        return false;
      }
    }
    return true;
  }

  bool Verifier::match(std::vector<Node*> left,
      std::vector<Node*> right,
      std::list<std::pair<Node*, Node*>> &matches) const {
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
          matches.push_back(std::make_pair(lNode, rNode));
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

  void Verifier::to_expr(Graph *graph,
    context &ctx,
    expr_vector nodes) const {

    std::cout << "Create equations for channels: " + graph->name << "\n";
    std::vector<Chan*> gChannels = graph->chans;

    for (const auto &channel : gChannels) {

      expr src = toConst(channel->source, ctx);
      expr tgt = toConst(channel->target, ctx);

      expr chanExpr = src == tgt;
      nodes.push_back(chanExpr);
    }

    std::cout << "Create equations for nodes: " + graph->name << "\n";
    std::vector<Node*> gNodes = graph->nodes;

    for (const auto &node : gNodes) {

      if (node->is_delay()) {

        // treat delay node as in-to-out channel
        const Binding input = node->inputs[0]->target;
        const Binding output = node->outputs[0]->source;

        expr in = toConst(input, ctx);
        expr out = toConst(output, ctx);

        expr delayExpr = in == out;

        nodes.push_back(delayExpr);

      } else if (node->is_kernel()) {

        std::string funcName = node->name;
        std::vector<Chan*> nodeOuts = node->outputs;

        for (const auto &nodeOut : nodeOuts) {

          const Port *outPort = nodeOut->source.port;
          std::string funcIdxName = nodeOut->name;
          std::string modelName = nodeOut->graph.model.name;
          sort_vector sorts = getInSorts(node, ctx);
          std::string kerName = funcName + "_" + funcIdxName + "_" + modelName;
          const char *sortName =outPort->type.c_str();
          sort fSort = ctx.uninterpreted_sort(sortName);
          func_decl kernelFunc = function(kerName.c_str(), sorts, fSort);
          const expr_vector kernelArgs = getArgs(node, ctx);
          std::string constName = outPort->type + "_" + modelName;
          expr nodeOutConst = ctx.constant(constName.c_str(), fSort);
          expr kernelEq = kernelFunc(kernelArgs) == nodeOutConst;

          nodes.push_back(kernelEq);
        }
      } else if (node->is_merge()) {

        // merge has the only output
        Chan *nodeOut = node->outputs[0];
        const char *outSortName = nodeOut->source.node->type.name.c_str();
        sort outSort = ctx.uninterpreted_sort(outSortName);

        expr outConst = toConst(nodeOut->source, ctx);

        expr_vector mergeVec(ctx);
        std::vector< Chan*> nodeInputs = node->inputs;

        for (const auto &nodeInput : nodeInputs) {

          expr inConst = toConst(nodeInput->target, ctx);

          mergeVec.push_back(outConst == inConst);
        }

        expr mergeExpr = mk_or(mergeVec);
        nodes.push_back(mergeExpr);

      } else if (node->is_split()) {

        // split has the only input
        Chan *nodeInput = node->inputs[0];
        expr inConst = toConst(nodeInput->target, ctx);
        expr_vector splitVec(ctx);
        std::vector< Chan*> nodeOutputs = node->outputs;

        for (const auto &nodeOut : nodeOutputs) {

          expr outConst = toConst(nodeOut->source, ctx);
          expr outEq = inConst == outConst;
          splitVec.push_back(outEq);
        }

        expr splitExpr = mk_or(splitVec);
        nodes.push_back(splitExpr);
      } else {
        // sink or source, do nothing
        assert(node->is_sink() || node->is_source());
      }
    }
  }

  std::vector<Node*> Verifier::getSources(Graph *graph) const {

    std::vector<Node*> result;
    std::vector<Node*> graphNodes = graph->nodes;

    for (const auto &node : graphNodes) {

      if (node->is_source()) {
        result.push_back(node);
      }
    }
    return result;
  }

  std::vector<Node*> Verifier::getSinks(Graph *graph) const {

    std::vector<Node*> result;
    std::vector<Node*> graphNodes = graph->nodes;

    for (const auto &node : graphNodes) {

      if (node->is_sink()) {
        result.push_back(node);
      }
    }
    return result;
  }

  sort Verifier::getSort(const Node *node, context &ctx) const {
    return ctx.uninterpreted_sort(node->type.name.c_str());
  }

  expr Verifier::toConst(const Binding &bnd, context &ctx) const {

    const Port *port = bnd.port;
    const char *typeName = port->type.c_str();
    sort fInSort = ctx.uninterpreted_sort(typeName);
    std::string modelName = bnd.node->graph.model.name;
    std::string constName = port->name + "_" + modelName;

    return ctx.constant(constName.c_str(), fInSort);
  }

  expr Verifier::toConst(const Node *node, context &ctx) const {

    const NodeType &fType = node->type;
    std::string typeName = fType.name;
    sort fInSort = ctx.uninterpreted_sort(typeName.c_str());
    std::string modelName = node->graph.model.name;
    std::string constName = node->name + "_" + modelName;

    return ctx.constant(constName.c_str(), fInSort);
  }

  expr Verifier::toFunc(const Node *node, const Chan *ch, context &ctx) const {

    const Binding src = ch->source;
    const Port *fPort = src.port;
    std::string outIdx = ch->name;
    sort_vector sorts(ctx);
    std::string modelName = node->graph.model.name;
    std::string funcName = node->name + "_" + outIdx + "_" + modelName;
    const char *sortName = fPort->type.c_str();
    sort fSort = ctx.uninterpreted_sort(sortName);
    func_decl func = function(funcName.c_str(), sorts, fSort);
    const expr_vector fArgs = expr_vector(ctx);
    return func(fArgs);
  }

  sort_vector Verifier::getInSorts(const Node *node, context &ctx) const {
    unsigned arity = node->inputs.size();
    sort_vector sorts(ctx);

    for (size_t i = 0; i < arity; i++) {

      const char *sortName = node->inputs[i]->target.port->type.c_str();
      sorts.push_back(ctx.uninterpreted_sort(sortName));
    }
    return sorts;
  }

  expr_vector Verifier::getArgs(const Node *node, context &ctx) const {

    std::vector<Chan*> inputs = node->inputs;
    unsigned arity = inputs.size();

    expr_vector args(ctx);

    for (size_t i = 0; i < arity; i++) {
      args.push_back(toConst(inputs[i]->target, ctx));
    }

    return args;
  }
} // namespace eda::hls::debugger
