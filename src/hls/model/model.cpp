//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/model/model.h>

namespace eda::hls::model {

std::ostream& operator <<(std::ostream &out, const Port &port) {
  out << port.type << "<" << port.flow << ">" << " ";
  out << "#" << port.latency << " " << port.name;
  if (port.is_const) {
    out << "=" << port.value;
  }
  return out;
}

static std::ostream& operator <<(std::ostream &out, const std::vector<Port *> &ports) {
  bool comma = false;
  for (const Port *port: ports) {
    out << (comma ? ", " : "") << *port;
    comma = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const NodeType &nodetype) {
  out << "  nodetype " << nodetype.name;
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
  out << "  graph " << graph.name << " {" << std::endl;

  for (const Chan *chan: graph.chans) {
    out << "    " << *chan << std::endl;
  }

  for (const Node *node: graph.nodes) {
    out << "    " << *node << std::endl;
  }

  return out << "  }";
}

std::ostream& operator <<(std::ostream &out, const Model &model) {
  out << "model " << model.name << "{" << std::endl;

  for (const NodeType *nodetype: model.nodetypes) {
    out << *nodetype << std::endl;
  }

  for (const Graph *graph: model.graphs) {
    out << *graph << std::endl;
  }

  return out << "}";
}

} // namespace eda::hls::model
