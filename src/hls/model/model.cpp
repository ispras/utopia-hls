//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/model/model.h>

namespace eda::hls::model {

std::ostream& operator <<(std::ostream &out, const Argument &argument) {
  out << argument.type << "<" << argument.flow << ">" << " ";
  return out << "#" << argument.latency << " " << argument.name;
}

static std::ostream& operator <<(std::ostream &out, const std::vector<Argument *> &args) {
  bool comma = false;
  for (const Argument *argument: args) {
    out << (comma ? ", " : "") << *argument;
    comma = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const NodeType &nodetype) {
  out << "nodetype " << nodetype.name;
  return out << "(" << nodetype.inputs << ") => (" << nodetype.outputs << ");";
}

std::ostream& operator <<(std::ostream &out, const Chan &chan) {
  return out << "chan " << chan.type << " " << chan.name << ";";
}

static std::ostream& operator <<(std::ostream &out, const std::vector<Chan *> &params) {
  bool comma = false;
  for (const Chan *chan: params) {
    out << (comma ? ", " : "") << chan->name;
    comma = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const Node &node) {
  out << "node " << node.type.name << " " << node.name;
  return out << "(" << node.inputs << ") => (" << node.outputs << ");";
}

std::ostream& operator <<(std::ostream &out, const Graph &graph) {
  out << "graph " << graph.name << " {" << std::endl;

  for (const Chan *chan: graph.chans) {
    out << "  " << *chan << std::endl;
  }

  for (const Node *node: graph.nodes) {
    out << "  " << *node << std::endl;
  }

  return out << "}";
}

std::ostream& operator <<(std::ostream &out, const Model &model) {
  for (const NodeType *nodetype: model.nodetypes) {
    out << *nodetype << std::endl;
  }

  for (const Graph *graph: model.graphs) {
    out << *graph << std::endl;
  }

  return out;
}

} // namespace eda::hls::model
