//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "util/string.h"

using namespace eda::utils;

namespace eda::hls::model {

struct Model;
struct Graph;
struct Node;

struct Port final {
  Port(
    const std::string &name,
    const std::string &type,
    float flow,
    unsigned latency,
    bool isConst,
    unsigned value):
      name(name),
      type(type),
      flow(flow),
      latency(latency),
      isConst(isConst),
      value(value) {}

  const std::string name;
  const std::string type;
  const float flow;
  const unsigned latency;
  const bool isConst;
  const unsigned value;
};

struct NodeType final {
  NodeType(const std::string &name, const Model &model):
    name(name), model(model) {}

  void addInput(Port *input) {
    inputs.push_back(input);
  }

  void addOutput(Port *output) {
    outputs.push_back(output);
  }

  bool isConst() const {
    if (!inputs.empty())
      return false;

    for (const auto *output: outputs) {
      if (!output->isConst)
        return false;
    }

    return true;
  }

  bool isSource() const {
    return inputs.empty() && !isConst();
  }

  bool isSink() const {
    return outputs.empty();
  }

  bool isMerge() const {
    return outputs.size() == 1
        && starts_with(name, "merge");
  }

  bool isSplit() const {
    return inputs.size() == 1
        && starts_with(name, "split");
  }

  bool isDup() const {
    return inputs.size() == 1
        && starts_with(name, "dup");
  }

  bool isDelay() const {
    return inputs.size() == 1
        && outputs.size() == 1
        && starts_with(name, "delay");
  }

  bool isKernel() const {
    return !isConst()
        && !isSource()
        && !isSink()
        && !isMerge()
        && !isSplit()
        && !isDup()
        && !isDelay();
  }

  const std::string name;
  std::vector<Port*> inputs;
  std::vector<Port*> outputs;

  // Reference to the parent.
  const Model &model;
};

struct Binding final {
  Binding():
    node(nullptr), port(nullptr) {}
  Binding(const Node *node, const Port *port):
    node(node), port(port) {}

  bool isLinked() const {
    return node != nullptr;
  }

  const Node *node;
  const Port *port;
};

struct Chan final {
  Chan(const std::string &name, const std::string &type, const Graph &graph):
    name(name), type(type), graph(graph) {}

  const std::string name;
  const std::string type;

  Binding source;
  Binding target;

  // Reference to the parent.
  const Graph &graph;
};

struct Node final {
  Node(const std::string &name, const NodeType &type, const Graph &graph):
    name(name), type(type), graph(graph) {}

  void addInput(Chan *input) {
    inputs.push_back(input);
  }

  void addOutput(Chan *output) {
    outputs.push_back(output);
  }

  bool isConst()  const { return type.isConst();  }
  bool isSource() const { return type.isSource(); }
  bool isSink()   const { return type.isSink();   }
  bool isMerge()  const { return type.isMerge();  }
  bool isSplit()  const { return type.isSplit();  }
  bool isDup()    const { return type.isDup();    }
  bool isDelay()  const { return type.isDelay();  }
  bool isKernel() const { return type.isKernel(); }

  const std::string name;
  const NodeType &type;
  std::vector<Chan*> inputs;
  std::vector<Chan*> outputs;

  // Reference to the parent.
  const Graph &graph;
};

struct Graph final {
  Graph(const std::string &name, const Model &model):
    name(name), model(model) {}

  void addChan(Chan *chan) {
    chans.push_back(chan);
  }

  void addNode(Node *node) {
    nodes.push_back(node);
  }

  void instantiate(
    const Graph &graph,
    const std::string &name,
    const std::map<std::string, std::map<std::string, Chan*>> &inputs,
    const std::map<std::string, std::map<std::string, Chan*>> &outputs);

  const std::string name;
  std::vector<Chan*> chans;
  std::vector<Node*> nodes;

  // Reference to the parent.
  const Model &model;
};

struct Model final {
  Model(const std::string &name):
    name(name) {}

  void addNodetype(NodeType *nodetype) {
    nodetypes.push_back(nodetype);
  }

  void addGraph(Graph *graph) {
    graphs.push_back(graph);
  }

  const std::string name;
  std::vector<NodeType *> nodetypes;
  std::vector<Graph *> graphs;
};

std::ostream& operator <<(std::ostream &out, const Port &port);
std::ostream& operator <<(std::ostream &out, const NodeType &nodetype);
std::ostream& operator <<(std::ostream &out, const Chan &chan);
std::ostream& operator <<(std::ostream &out, const Node &node);
std::ostream& operator <<(std::ostream &out, const Graph &graph);
std::ostream& operator <<(std::ostream &out, const Model &model);

} // namespace eda::hls::model
