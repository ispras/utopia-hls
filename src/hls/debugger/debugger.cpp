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

      // TODO: add input bindings to nodes
      // TODO: add output bindings to nodes
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

  z3::func_decl Verifier::mkFunction(const char *name, sort fSort) const {
    return function(name, fSort, fSort);
  }

  void Verifier::to_expr(Graph *graph,
    context &ctx,
    std::vector<expr*> nodes) const {

    // create equations for channels
    std::vector<Chan*> g_channels = graph->chans;

    for (const auto &channel : g_channels) {

      const char *srcName = (channel->source->name).c_str();
      const char *tgtName = (channel->target->name).c_str();

      sort fSort = ctx.uninterpreted_sort(channel->type.c_str());
      func_decl src_func = mkFunction(srcName, fSort);
      func_decl tgt_func = mkFunction(tgtName, fSort);

      expr ch_expr = src_func() == tgt_func();
      nodes.push_back(&ch_expr);
    }

    // create equations for nodes
    std::vector<Node*> g_nodes = graph->nodes;

    for (const auto &node : g_nodes) {

      sort fSort = ctx.uninterpreted_sort(node->type.name.c_str());

      if (node->is_delay()) {

        // treat delay node as in-to-out channel
        std::string input = node->inputs[0]->target->name;
        std::string output = node->outputs[0]->source->name;

        func_decl in_func = mkFunction(input.c_str(), fSort);
        func_decl out_func = mkFunction(output.c_str(), fSort);

        expr delay_expr = in_func() == out_func();

        nodes.push_back(&delay_expr);

      } else if (node->is_kernel()) {

      } else if (node->is_merge()) {

      } else if (node->is_split()) {

      } else {
        // sink or source, do nothing
        assert(node->is_sink() || node->is_source());
      }
    }
  }
} // namespace eda::hls::debugger