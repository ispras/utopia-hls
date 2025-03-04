//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

using namespace dfcxx; // For testing purposes only.

#define STUB_NAME "Name"

TEST(DFCXXGraph, AddNode) {
  const auto direction = DFVariableImpl::IODirection::NONE;
  const auto opType = OpType::ADD;
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  meta.storage.addType(type);
  DFVariableImpl *var = meta.varBuilder.buildStream("", direction, &meta, type);
  meta.storage.addVariable(var);
  auto result = meta.graph.addNode(var, opType, NodeData {});
  EXPECT_TRUE(result.first);
  EXPECT_TRUE(result.second);
  const Nodes &nodes = meta.graph.getNodes();
  const Nodes &startNodes = meta.graph.getStartNodes();
  EXPECT_EQ(nodes.size(), 1);
  EXPECT_EQ(startNodes.size(), 0);
  Node *node = *(nodes.begin());
  EXPECT_EQ(node->var, var);
  EXPECT_EQ(node->type, opType);
  EXPECT_EQ(result.first, node);
  const NodeNameMap &map = meta.graph.getNameMap();
  EXPECT_EQ(map.size(), 0);
}

TEST(DFCXXGraph, AddStartNode) {
  const auto direction = DFVariableImpl::IODirection::INPUT;
  const auto opType = OpType::IN;
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  meta.storage.addType(type);
  DFVariableImpl *var = meta.varBuilder.buildStream(STUB_NAME, direction, &meta, type);
  meta.storage.addVariable(var);
  auto result = meta.graph.addNode(var, opType, NodeData {});
  EXPECT_TRUE(result.first);
  EXPECT_TRUE(result.second);
  const Nodes &nodes = meta.graph.getNodes();
  const Nodes &startNodes = meta.graph.getStartNodes();
  EXPECT_EQ(nodes.size(), 1);
  EXPECT_EQ(startNodes.size(), 1);
  Node *node = *(nodes.begin());
  Node *startNode = *(startNodes.begin());
  EXPECT_EQ(node, startNode);
  EXPECT_EQ(node->var, var);
  EXPECT_EQ(node->type, opType);
  EXPECT_EQ(result.first, node);
  const NodeNameMap &map = meta.graph.getNameMap();
  EXPECT_EQ(map.size(), 1);
  EXPECT_EQ(map.at(STUB_NAME), node);
}

TEST(DFCXXGraph, AddAlreadyExistingNode) {
  const auto direction = DFVariableImpl::IODirection::NONE;
  const auto opType = OpType::ADD;
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  meta.storage.addType(type);
  DFVariableImpl *var = meta.varBuilder.buildStream("", direction, &meta, type);
  meta.storage.addVariable(var);
  auto result = meta.graph.addNode(var, opType, NodeData {});
  result = meta.graph.addNode(var, opType, NodeData {});
  EXPECT_TRUE(result.first);
  EXPECT_TRUE(!result.second);
  const Nodes &nodes = meta.graph.getNodes();
  const Nodes &startNodes = meta.graph.getStartNodes();
  EXPECT_EQ(nodes.size(), 1);
  EXPECT_EQ(startNodes.size(), 0);
  Node *node = *(nodes.begin());
  EXPECT_EQ(node->var, var);
  EXPECT_EQ(node->type, opType);
  EXPECT_EQ(result.first, node);
  const NodeNameMap &map = meta.graph.getNameMap();
  EXPECT_EQ(map.size(), 0);
}

TEST(DFCXXGraph, FindNodeByName) {
  const auto direction = DFVariableImpl::IODirection::INPUT;
  const auto opType = OpType::IN;
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  meta.storage.addType(type);

  DFVariableImpl *var1 = meta.varBuilder.buildStream(STUB_NAME "1", direction, &meta, type);
  meta.storage.addVariable(var1);
  auto result1 = meta.graph.addNode(var1, opType, NodeData {});
  EXPECT_TRUE(result1.first);
  EXPECT_TRUE(result1.second);

  DFVariableImpl *var2 = meta.varBuilder.buildStream(STUB_NAME "2", direction, &meta, type);
  meta.storage.addVariable(var2);
  auto result2 = meta.graph.addNode(var2, opType, NodeData {});
  EXPECT_TRUE(result2.first);
  EXPECT_TRUE(result2.second);

  DFVariableImpl *var3 = meta.varBuilder.buildStream(STUB_NAME "3", direction, &meta, type);
  meta.storage.addVariable(var3);
  auto result3 = meta.graph.addNode(var3, opType, NodeData {});
  EXPECT_TRUE(result3.first);
  EXPECT_TRUE(result3.second);

  EXPECT_EQ(meta.graph.getNameMap().size(), 3);
  EXPECT_EQ(meta.graph.findNode(STUB_NAME "2"), result2.first);
}

TEST(DFCXXGraph, FindNode) {
  const auto direction = DFVariableImpl::IODirection::INPUT;
  const auto opType = OpType::IN;
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  meta.storage.addType(type);

  DFVariableImpl *var1 = meta.varBuilder.buildStream(STUB_NAME "1", direction, &meta, type);
  meta.storage.addVariable(var1);
  auto result1 = meta.graph.addNode(var1, opType, NodeData {});
  EXPECT_TRUE(result1.first);
  EXPECT_TRUE(result1.second);

  DFVariableImpl *var2 = meta.varBuilder.buildStream(STUB_NAME "2", direction, &meta, type);
  meta.storage.addVariable(var2);
  auto result2 = meta.graph.addNode(var2, opType, NodeData {});
  EXPECT_TRUE(result2.first);
  EXPECT_TRUE(result2.second);

  DFVariableImpl *var3 = meta.varBuilder.buildStream(STUB_NAME "3", direction, &meta, type);
  meta.storage.addVariable(var3);
  auto result3 = meta.graph.addNode(var3, opType, NodeData {});
  EXPECT_TRUE(result3.first);
  EXPECT_TRUE(result3.second);

  EXPECT_EQ(meta.graph.findNode(var2), result2.first);
}