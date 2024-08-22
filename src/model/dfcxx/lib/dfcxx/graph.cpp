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

Node Graph::findNode(DFVariableImpl *var) {
  return *std::find_if(nodes.begin(), nodes.end(),
                       [&](const Node &node) { return node.var == var; });
}

void Graph::addNode(DFVariableImpl *var, OpType type, NodeData data) {
  nodes.emplace(var, type, data);
  if (type == IN || type == CONST) {
    startNodes.emplace(var, type, data);
  }
}

void Graph::addNode(const DFVariable &var, OpType type, NodeData data) {
  addNode(var.getImpl(), type, data);
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

void Graph::addChannel(const DFVariable &source, const DFVariable &target,
                       unsigned opInd, bool connect) {
  addChannel(source.getImpl(), target.getImpl(), opInd, connect);
}

} // namespace dfcxx
