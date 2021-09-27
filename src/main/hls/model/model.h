/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#pragma once

#include <iostream>
#include <string>
#include <vector>

namespace eda::hls::model {

struct ChanType final {
  ChanType(const std::string &name, const std::string &datatype, float maxflow):
    name(name), datatype(datatype), maxflow(maxflow) {}

  std::string name;
  std::string datatype;
  float maxflow;
};

struct NodeType final {
  NodeType(const std::string &name, unsigned latency):
    name(name), latency(latency) {}

  void add_input(ChanType *type) {
    inputs.push_back(type);
  }

  void add_output(ChanType *type) {
    outputs.push_back(type);
  }

  std::string name;
  unsigned latency;
  std::vector<ChanType *> inputs;
  std::vector<ChanType *> outputs;
};

struct Chan final {
  Chan(const std::string &name, const std::string &datatype):
    name(name), datatype(datatype) {}

  std::string name;
  std::string datatype;
  const ChanType *source = nullptr;
  const ChanType *target = nullptr;
};

struct Node final {
  Node(const NodeType &type):
    type(type) {}

  void add_input(Chan *chan) {
    inputs.push_back(chan);
  }

  void add_output(Chan *chan) {
    outputs.push_back(chan);
  }

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

  void add_chantype(ChanType *type) {
    chantypes.push_back(type);
  }

  void add_nodetype(NodeType *type) {
    nodetypes.push_back(type);
  }

  void add_graph(Graph *graph) {
    graphs.push_back(graph);
  }

  std::vector<ChanType *> chantypes;
  std::vector<NodeType *> nodetypes;
  std::vector<Graph *> graphs;
};

std::ostream& operator <<(std::ostream &out, const ChanType &chantype);
std::ostream& operator <<(std::ostream &out, const NodeType &nodetype);
std::ostream& operator <<(std::ostream &out, const Chan &chan);
std::ostream& operator <<(std::ostream &out, const Node &node);
std::ostream& operator <<(std::ostream &out, const Graph &graph);
std::ostream& operator <<(std::ostream &out, const Model &model);

} // namespace eda::hls::model
