//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/graph.h"

#include <algorithm>

namespace dfcxx {

const std::unordered_set<Node> &Graph::getNodes() const {
  return nodes;
}

const std::unordered_set<Node> &Graph::getStartNodes() const {
  return startNodes;
}

const std::unordered_map<Node, std::vector<Channel>> &Graph::getInputs() const {
  return inputs;
}

const std::unordered_map<Node, std::vector<Channel>> &Graph::getOutputs() const {
  return outputs;
}

const std::unordered_map<Node, Channel> &Graph::getConnections() const {
  return connections;
}

Node Graph::findNode(DFVariableImpl *var) {
  return *std::find_if(nodes.begin(), nodes.end(),
                       [&](const Node &node) { return node.var == var; });
}

void Graph::addNode(DFVariableImpl *var, OpType type, NodeData data) {
  auto node = nodes.emplace(var, type, data);
  if (type == IN || type == CONST) {
    startNodes.emplace(var, type, data);
  }
  // The following lines create empty channel vectors
  // for new nodes. This allows to use .at() on unconnected
  // nodes without getting an exception.
  (void)inputs[*(node.first)];
  (void)outputs[*(node.first)];
}

void Graph::addChannel(DFVariableImpl *source, DFVariableImpl *target,
                       unsigned opInd, bool connect) {
  Node foundSource = findNode(source);
  Node foundTarget = findNode(target);
  Channel newChannel(foundSource, foundTarget, opInd);
  outputs[foundSource].push_back(newChannel);
  inputs[foundTarget].push_back(newChannel);
  if (connect) {
    connections.insert(std::make_pair(foundTarget, newChannel));
    connections.at(foundTarget) = newChannel;
  }
}

} // namespace dfcxx
