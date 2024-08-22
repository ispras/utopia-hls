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

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dfcxx {

class Graph {
private:
  std::unordered_set<Node> nodes;
  std::unordered_set<Node> startNodes;
  std::unordered_map<Node, std::vector<Channel>> inputs;
  std::unordered_map<Node, std::vector<Channel>> outputs;
  std::unordered_map<Node, Channel> connections;

  Graph() = default;

  Node findNode(DFVariableImpl *var);

  void addNode(DFVariableImpl *var, OpType type, NodeData data);

  void addNode(const DFVariable &var, OpType type, NodeData data);

  void addChannel(DFVariableImpl *source, DFVariableImpl *target,
                  unsigned opInd, bool connect);

  void addChannel(const DFVariable &source, const DFVariable &target,
                  unsigned opInd, bool connect);
};

} // namespace dfcxx

#endif // DFCXX_GRAPH_H
