//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "util/string.h"

using namespace eda::utils;

namespace eda::hls::model {

struct Argument final {
  Argument(const std::string &name, const std::string &type, float flow, unsigned latency):
    name(name), type(type), flow(flow), latency(latency) {}

  std::string name;
  std::string type;
  float flow;
  unsigned latency;
};

struct NodeType final {
  NodeType(const std::string &name):
    name(name) {}

  void add_input(Argument *input) {
    inputs.push_back(input);
  }

  void add_output(Argument *output) {
    outputs.push_back(output);
  }

  bool is_source() const {
    return inputs.empty();
  }

  bool is_sink() const {
    return outputs.empty();
  }

  bool is_merge() const {
    return outputs.size() == 1
        && starts_with(name, "merge");
  }

  bool is_split() const {
    return inputs.size() == 1
        && starts_with(name, "split");
  }

  bool is_delay() const {
    return inputs.size() == 1 && outputs.size() == 1
        && starts_with(name, "delay");
  }

  bool is_kernel() const {
    return !is_source()
        && !is_sink()
        && !is_merge()
        && !is_split()
        && !is_delay();
  }

  std::string name;
  std::vector<Argument *> inputs;
  std::vector<Argument *> outputs;
};

struct Node;

struct Binding final {
  Binding():
    node(nullptr), port(nullptr) {}
  Binding(const Node *node, const Argument *port):
    node(node), port(port) {}

  bool is_linked() const {
    return node != nullptr;
  }

  const Node *node;
  const Argument *port;
};

struct Chan final {
  Chan(const std::string &name, const std::string &type):
    name(name), type(type) {}

  std::string name;
  std::string type;
  Binding source;
  Binding target;
};

struct Node final {
  Node(const std::string &name, const NodeType &type):
    name(name), type(type) {}

  void add_input(Chan *input) {
    inputs.push_back(input);
  }

  void add_output(Chan *output) {
    outputs.push_back(output);
  }

  bool is_source() const { return type.is_source(); }
  bool is_sink()   const { return type.is_sink();   }
  bool is_merge()  const { return type.is_merge();  }
  bool is_split()  const { return type.is_split();  }
  bool is_delay()  const { return type.is_delay();  }
  bool is_kernel() const { return type.is_kernel(); }

  std::string name;
  const NodeType &type;
  std::vector<Chan *> inputs;
  std::vector<Chan *> outputs;
};

struct Graph final {
  Graph(const std::string &name):
    name(name) {}

  void add_chan(Chan *chan) {
    chans.push_back(chan);
  }

  void add_node(Node *node) {
    nodes.push_back(node);
  }

  std::string name;
  std::vector<Chan *> chans;
  std::vector<Node *> nodes;
};

struct Model final {
  Model() = default;

  void add_nodetype(NodeType *nodetype) {
    nodetypes.push_back(nodetype);
  }

  void add_graph(Graph *graph) {
    graphs.push_back(graph);
  }

  std::vector<NodeType *> nodetypes;
  std::vector<Graph *> graphs;
};

std::ostream& operator <<(std::ostream &out, const Argument &argument);
std::ostream& operator <<(std::ostream &out, const NodeType &nodetype);
std::ostream& operator <<(std::ostream &out, const Chan &chan);
std::ostream& operator <<(std::ostream &out, const Node &node);
std::ostream& operator <<(std::ostream &out, const Graph &graph);
std::ostream& operator <<(std::ostream &out, const Model &model);

} // namespace eda::hls::model
