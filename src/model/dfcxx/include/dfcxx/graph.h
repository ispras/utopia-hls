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
#include "dfcxx/typebuilders/builder.h"
#include "dfcxx/vars/var.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dfcxx {

class GraphHelper;

class Kernel;

class IO;

class Offset;

class Constant;

class Control;

class DFCIRBuilder;

class VarBuilder;

class KernStorage;

class Graph {
  friend GraphHelper;
  friend Kernel;
  friend IO;
  friend Offset;
  friend Constant;
  friend Control;
  friend DFCIRBuilder;
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

  void addChannel(DFVariableImpl *source, DFVariableImpl *target, unsigned opInd,
                  bool connect);

  void addChannel(const DFVariable &source, const DFVariable &target, unsigned opInd,
                  bool connect);
};

class GraphHelper {
  friend IO;
  friend Offset;
  friend Constant;
  friend Control;
private:
  Graph &graph;

  GraphHelper(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
              KernStorage &storage);

public:
  TypeBuilder &typeBuilder;
  VarBuilder &varBuilder;
  KernStorage &storage;

  void addNode(DFVariableImpl *var, OpType type, NodeData data);

  void addNode(const DFVariable &var, OpType type, NodeData data);

  void addChannel(DFVariableImpl *source, DFVariableImpl *target, unsigned opInd,
                  bool connect);

  void addChannel(const DFVariable &source, const DFVariable &target, unsigned opInd,
                  bool connect);
};

} // namespace dfcxx

#endif // DFCXX_GRAPH_H
