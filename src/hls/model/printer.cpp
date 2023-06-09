//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/printer.h"

namespace eda::hls::model {

static void printDot(std::ostream &out, const Node &node) {
  out << "    " << node.name << " ["
                << "label=" << node.type.name << ", "
                << "color=" << (node.isDelay() ? "red" : "black")
                << "];" << std::endl;
}

static void printDot(std::ostream &out, const Chan &chan) {
  if (chan.source.node != nullptr && chan.target.node != nullptr) {
    out << "    " << chan.source.node->name << " -> "
                  << chan.target.node->name << ";" << std::endl;
  }
}

static void printDot(std::ostream &out, const Graph &graph) {
  out << "  subgraph " << graph.name << " {" << std::endl;

  for (const auto *node : graph.nodes)
    printDot(out, *node);

  out << std::endl;

  for (const auto *chan : graph.chans)
    printDot(out, *chan);

  out << "  }" << std::endl;
}

void printDot(std::ostream &out, const Model &model) {
  out << "digraph " << model.name << " {" << std::endl;

  for (const auto *graph : model.graphs)
    printDot(out, *graph);

  out << "}" << std::endl;
}

} // namespace eda::hls::model