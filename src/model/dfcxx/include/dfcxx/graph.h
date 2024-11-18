//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_GRAPH_H
#define DFCXX_GRAPH_H

#include "dfcxx/channel.h"
#include "dfcxx/node.h"
#include "dfcxx/vars/var.h"

#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dfcxx {

class Graph {
private:
  std::unordered_map<std::string_view, Node> nameMap;
  std::unordered_set<Node> nodes;
  std::unordered_set<Node> startNodes;
  std::unordered_map<Node, std::vector<Channel>> inputs;
  std::unordered_map<Node, std::vector<Channel>> outputs;
  std::unordered_map<Node, Channel> connections;

public:
  const std::unordered_set<Node> &getNodes() const;

  const std::unordered_set<Node> &getStartNodes() const;

  const std::unordered_map<Node, std::vector<Channel>> &getInputs() const;

  const std::unordered_map<Node, std::vector<Channel>> &getOutputs() const;

  const std::unordered_map<Node, Channel> &getConnections() const;

  Node findNode(DFVariableImpl *var);

  void addNode(DFVariableImpl *var, OpType type, NodeData data);

  void addChannel(DFVariableImpl *source, DFVariableImpl *target,
                  unsigned opInd, bool connect);

  void transferFrom(Graph &&graph);

  Node getNodeByName(const std::string &name);

  void resetNodeName(const std::string &name);

  void deleteNode(Node node);

  void rebindInput(Node source, Node input, Graph &graph);

  Node rebindOutput(Node output, Node target, Graph &graph);
};

} // namespace dfcxx

#endif // DFCXX_GRAPH_H
