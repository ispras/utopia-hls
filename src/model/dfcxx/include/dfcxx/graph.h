//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
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

struct NodePtrEq {
  size_t operator()(Node *left, Node *right) const noexcept {
    return *left == *right;
  }
};

struct ChannelPtrHash {
  size_t operator()(Channel *channel) const noexcept {
    return std::hash<Channel>()(*channel);
  }
};

struct ChannelPtrEq {
  size_t operator()(Channel *left, Channel *right) const noexcept {
    return *left == *right;
  }
};

typedef std::unordered_set<Node *, NodePtrHash, NodePtrEq> Nodes;
typedef std::unordered_map<std::string_view, Node *> NodeNameMap;

struct KernelMeta;

class Graph {
  using NodePtrPred = std::function<bool(Node *)>;

private:
  Nodes nodes;
  std::vector<ModuleInst> instances;

  NodeNameMap nameMap;
  Nodes startNodes;

public:
  Graph() = default;

  Graph(const Graph &) = default;

  ~Graph();

  const std::vector<ModuleInst> &getInstances() const {
    return instances;
  }

  const Nodes &getNodes() const { return nodes; }

  const NodeNameMap &getNameMap() const  { return nameMap; }

  const Nodes &getStartNodes() const { return startNodes; }

  std::pair<Node *, bool>
  addNode(DFVariableImpl *var, ModuleInst *inst, OpType type, NodeData data);

  Channel *addChannel(Node *source, Node *target,
                      unsigned opInd, bool connect);

  Channel *addChannel(DFVariableImpl *source, ModuleInst *sourceInst,
                      DFVariableImpl *target, ModuleInst *targetInst,
                      unsigned opInd, bool connect);

  ModuleInst *addInst(std::string name,
                      std::vector<ModuleInst::Port> ports,
                      std::vector<ModuleParam> params);

  void transferFrom(Graph &&graph);

  void resetNodeName(const std::string &name);

  void deleteNode(Node *node);

  void rebindInput(Node *source, Node *input);

  Node *rebindOutput(Node *output, Node *target);

  Node *findNode(const std::string &name);

  Node *findNode(DFVariableImpl *var, ModuleInst *inst);

  Node *findNodeIf(NodePtrPred pred);

  Node *findStartNode(DFVariableImpl *var);

  Node *findStartNodeIf(NodePtrPred pred);

  void resetMeta(KernelMeta *meta);
};

} // namespace dfcxx

#endif // DFCXX_GRAPH_H
