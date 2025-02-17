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

#include <functional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dfcxx {

struct NodePtrHash {
  size_t operator()(Node *node) const noexcept {
    return std::hash<Node>()(*node);
  }
};
  
struct ChannelPtrHash {
  size_t operator()(Channel *channel) const noexcept {
    return std::hash<Channel>()(*channel);
  }
};

typedef std::unordered_set<Node *, NodePtrHash> Nodes;
typedef std::unordered_set<Channel *, ChannelPtrHash> Channels;
typedef std::unordered_map<std::string_view, Node *> NodeNameMap;
typedef std::unordered_map<Node *, std::vector<Channel *>> ChannelMap;
typedef std::unordered_map<Node *, Channel *> ConnectionMap;

class Graph {
  using NodePtrPred = std::function<bool(Node *)>;

private:
  Nodes nodes;
  Channels channels;

  NodeNameMap nameMap;
  Nodes startNodes;
  ChannelMap inputs;
  ChannelMap outputs;
  ConnectionMap connections;

public:
  Graph() = default;

  Graph(const Graph &) = default;

  ~Graph();

  const Nodes &getNodes() const { return nodes; }

  const Channels &getChannels() const { return channels; }

  const NodeNameMap &getNameMap() const  { return nameMap; }

  const Nodes &getStartNodes() const { return startNodes; }

  const ChannelMap &getInputs() const { return inputs;}

  const ChannelMap &getOutputs() const { return outputs; }

  const ConnectionMap &getConnections() const { return connections; }

  std::pair<Node *, bool> addNode(DFVariableImpl *var, OpType type, NodeData data);

  Channel *addChannel(Node *source, Node *target,
                      unsigned opInd, bool connect);

  Channel *addChannel(DFVariableImpl *source, DFVariableImpl *target,
                      unsigned opInd, bool connect);

  void transferFrom(Graph &&graph);

  void resetNodeName(const std::string &name);

  void deleteNode(Node *node);

  void rebindInput(Node *source, Node *input, Graph &graph);

  Node *rebindOutput(Node *output, Node *target, Graph &graph);

  Node *findNode(const std::string &name);

  Node *findNode(DFVariableImpl *var);

  Node *findNodeIf(NodePtrPred pred);

  Node *findStartNode(DFVariableImpl *var);

  Node *findStartNodeIf(NodePtrPred pred);
};

} // namespace dfcxx

#endif // DFCXX_GRAPH_H
