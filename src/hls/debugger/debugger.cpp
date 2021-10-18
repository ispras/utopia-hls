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

    std::list<std::pair<Graph*, Graph*>> matchedGraphs;

    // If graphs are not completely matched
    if (!match(left.graphs, right.graphs, matchedGraphs))
      return false;

    context ctx;
    std::vector<expr *> nodes;

    for (const auto &pair : matchedGraphs) {

      Graph *fGraph = pair.first;
      Graph *sGraph = pair.second;

      to_expr(fGraph, ctx, nodes);
      to_expr(sGraph, ctx, nodes);

      // add input bindings to nodes
      std::vector<Node*> fInputs = getSources(fGraph);
      std::vector<Node*> sInputs = getSources(sGraph);
      std::list<std::pair<Node*, Node*>> inMatch;

      if (match(fInputs, sInputs, inMatch)) {

        for (const auto &inPair : inMatch) {

          Node *fIn = inPair.first;
          Node *sIn = inPair.second;

          const NodeType &fType = fIn->type;
          const NodeType &sType = sIn->type;

          assert(&fType == &sType);

          sort fSort = ctx.uninterpreted_sort(fIn->type.name.c_str());
          func_decl fFunc = mkFunction(fIn->name, fSort);
          func_decl sFunc = mkFunction(sIn->name, fSort);
          expr inEq = fFunc() == sFunc();

          nodes.push_back(&inEq);
        }
      } else {
        std::cout << "Cannot match graphs inputs" << std::endl;
      }

      // add output bindings to nodes
      std::vector<Node*> fOutputs = getSinks(fGraph);
      std::vector<Node*> sOutputs = getSinks(sGraph);
      std::list<std::pair<Node*, Node*>> outMatch;

      if (match(fOutputs, sOutputs, outMatch)) {
        for (const auto &outPair : outMatch) {

          Node *fOut = outPair.first;
          Node *sOut = outPair.second;

          const NodeType &fType = fOut->type;
          const NodeType &sType = sOut->type;

          assert(&fType == &sType);

          sort fSort = ctx.uninterpreted_sort(fOut->type.name.c_str());
          func_decl fFunc = mkFunction(fOut->name, fSort);
          func_decl sFunc = mkFunction(sOut->name, fSort);
          expr outEq = fFunc() == sFunc();

          nodes.push_back(&outEq);
        }
      } else {
        std::cout << "Cannot match graphs outputs" << std::endl;
      }
    }

    solver s(ctx);
    for (const auto &node : nodes) {
      s.add(*node);
    }

    switch (s.check()) {
      case sat:
        std::cout << "Models are equivalent" << std::endl;
        return true;
      case unsat:
        std::cout << "Models are not equivalent" << std::endl;
        return false;
      case unknown:
      default:
        std::cout << "Z3 solver says \"unknown\"" << std::endl;
        return false;
        break;
    }
  }

  bool Verifier::match(std::vector<Graph*> left,
      std::vector<Graph*> right,
      std::list<std::pair<Graph*, Graph*>> &matches) const {

    size_t lSize = left.size();
    size_t rSize = right.size();

    if (lSize != rSize)
      return false;

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
        std::cout << "No match for graphs " + lGraph->name << std:: endl;
        return false;
      }
    }
    return false;
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
        std::cout << "No match for graphs " + lNode->name << std:: endl;
        return false;
      }
    }
    return false;
  }

  func_decl Verifier::mkFunction(const std::string name, sort fSort) const {
    return function(name.c_str(), fSort, fSort);
  }

  func_decl Verifier::mkFunction(const std::string name, const Chan *channel, context &ctx) const {

    sort fSort = ctx.uninterpreted_sort(channel->type.c_str());
    return mkFunction(name, fSort);
  }

  void Verifier::to_expr(Graph *graph,
    context &ctx,
    std::vector<expr*> nodes) const {

    // create equations for channels
    std::vector<Chan*> gChannels = graph->chans;

    for (const auto &channel : gChannels) {

      std::string srcName = channel->source.node->name;
      std::string tgtName = channel->target.node->name;

      func_decl srcFunc = mkFunction(srcName, channel, ctx);
      func_decl tgtFunc = mkFunction(tgtName, channel, ctx);

      expr chanExpr = srcFunc() == tgtFunc();
      nodes.push_back(&chanExpr);
    }

    // create equations for nodes
    std::vector<Node*> gNodes = graph->nodes;

    for (const auto &node : gNodes) {

      sort fSort = ctx.uninterpreted_sort(node->type.name.c_str());

      if (node->is_delay()) {

        // treat delay node as in-to-out channel
        std::string input = node->inputs[0]->target.node->name;
        std::string output = node->outputs[0]->source.node->name;

        func_decl inFunc = mkFunction(input.c_str(), fSort);
        func_decl outFunc = mkFunction(output.c_str(), fSort);

        expr delayExpr = inFunc() == outFunc();

        nodes.push_back(&delayExpr);

      } else if (node->is_kernel()) {

        std::string funcName = node->name;

        std::vector<Chan*> nodeOuts = node->outputs;

        for (const auto &nodeOut : nodeOuts) {

          const char* funcIdxName = nodeOut->name.c_str();

          size_t arity = node->inputs.size();

          switch (arity) {

            case 1:
            {
              std::string arg0Name = node->inputs[0]->target.node->name;
              func_decl arg0 = mkFunction(arg0Name, node->inputs[0], ctx);
              func_decl kernelFunc = mkFunction(funcName + "_" + funcIdxName, fSort);
              func_decl val = mkFunction(funcIdxName, fSort);

              expr result = kernelFunc(arg0()) == val();
              nodes.push_back(&result);

              break;
            }
            case 2:
            {
              std::string arg0Name = node->inputs[0]->target.node->name;
              func_decl arg0 = mkFunction(arg0Name, node->inputs[0], ctx);

              std::string arg1Name = node->inputs[1]->target.node->name;
              func_decl arg1 = mkFunction(arg1Name, node->inputs[1], ctx);

              func_decl kernelFunc = mkFunction(funcName + "_" + funcIdxName, fSort);
              func_decl val = mkFunction(funcIdxName, fSort);

              expr result = kernelFunc(arg0(), arg1()) == val();
              nodes.push_back(&result);

              break;
            }
            case 3:
            {
              std::string arg0Name = node->inputs[0]->target.node->name;
              func_decl arg0 = mkFunction(arg0Name, node->inputs[0], ctx);

              std::string arg1Name = node->inputs[1]->target.node->name;
              func_decl arg1 = mkFunction(arg1Name, node->inputs[1], ctx);

              std::string arg2Name = node->inputs[2]->target.node->name;
              func_decl arg2 = mkFunction(arg2Name, node->inputs[2], ctx);

              func_decl kernelFunc = mkFunction(funcName + "_" + funcIdxName, fSort);
              func_decl val = mkFunction(funcIdxName, fSort);

              expr result = kernelFunc(arg0(), arg1(), arg2()) == val();
              nodes.push_back(&result);

              break;
            }
            case 4:
            {
              std::string arg0Name = node->inputs[0]->target.node->name;
              func_decl arg0 = mkFunction(arg0Name, node->inputs[0], ctx);

              std::string arg1Name = node->inputs[1]->target.node->name;
              func_decl arg1 = mkFunction(arg1Name, node->inputs[1], ctx);

              std::string arg2Name = node->inputs[2]->target.node->name;
              func_decl arg2 = mkFunction(arg2Name, node->inputs[2], ctx);

              std::string arg3Name = node->inputs[3]->target.node->name;
              func_decl arg3 = mkFunction(arg3Name, node->inputs[3], ctx);

              func_decl kernelFunc = mkFunction(funcName + "_" + funcIdxName, fSort);
              func_decl val = mkFunction(funcIdxName, fSort);

              expr result = kernelFunc(arg0(), arg1(), arg2(), arg3()) == val();
              nodes.push_back(&result);

              break;
            }
            case 5:
            {
              std::string arg0Name = node->inputs[0]->target.node->name;
              func_decl arg0 = mkFunction(arg0Name, node->inputs[0], ctx);

              std::string arg1Name = node->inputs[1]->target.node->name;
              func_decl arg1 = mkFunction(arg1Name, node->inputs[1], ctx);

              std::string arg2Name = node->inputs[2]->target.node->name;
              func_decl arg2 = mkFunction(arg2Name, node->inputs[2], ctx);

              std::string arg3Name = node->inputs[3]->target.node->name;
              func_decl arg3 = mkFunction(arg3Name, node->inputs[3], ctx);

              std::string arg4Name = node->inputs[4]->target.node->name;
              func_decl arg4 = mkFunction(arg4Name, node->inputs[4], ctx);

              func_decl kernelFunc = mkFunction(funcName + "_" + funcIdxName, fSort);
              func_decl val = mkFunction(funcIdxName, fSort);

              expr result = kernelFunc(arg0(), arg1(), arg2(), arg3()) == val();
              nodes.push_back(&result);

              break;
            }
            default:
              std::cout << "Unsupported arity of " + funcName + " func node: "
                  + std::to_string(arity) << std::endl;
              break;
          }
        }
      } else if (node->is_merge()) {

        // merge has the only output
        Chan *nodeOut = node->outputs[0];
        func_decl val = mkFunction(nodeOut->source.node->name, fSort);

        expr mergeExpr = expr(ctx);
        std::vector< Chan*> nodeInputs = node->inputs;

        for (const auto &nodeInput : nodeInputs) {

          std::string inputName = nodeInput->target.node->name;
          func_decl inputFunc = mkFunction(inputName, fSort);

          mergeExpr = mergeExpr || (inputFunc() == val());
        }

        nodes.push_back(&mergeExpr);

      } else if (node->is_split()) {

        //split has the only input
        Chan *nodeInput = node->inputs[0];
        func_decl arg = mkFunction(nodeInput->target.node->name, fSort);

        expr splitExpr = expr(ctx);
        std::vector< Chan*> nodeOutputs = node->outputs;

        for (const auto &nodeOut : nodeOutputs) {

          std::string outName = nodeOut->source.node->name;
          func_decl outFunc = mkFunction(outName, fSort);

          splitExpr = splitExpr || (arg() == outFunc());
        }

        nodes.push_back(&splitExpr);
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
} // namespace eda::hls::debugger
