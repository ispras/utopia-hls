//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
// Utility methods.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "HIL/Ops.h"
#include "mlir/Support/LLVM.h"

#include <optional>
#include <vector>

template <typename T, typename Iterator>
std::vector<T> findElemsByType(Iterator first, Iterator last) {
  std::vector<T> result;
  for (; first != last; ++first) {
    auto elem = mlir::dyn_cast<T>(*first);
    if (elem) {
      result.push_back(elem);
    }
  }
  return result;
}

template <typename T, typename Iterator>
std::optional<T> findElemByType(Iterator first, Iterator last) {
  for (; first != last; ++first) {
    auto elem = mlir::dyn_cast<T>(*first);
    if (elem) {
      return elem;
    }
  }
  return std::nullopt;
}

template <typename T, typename Container>
std::optional<T> findElemByType(Container &&c) {
  return findElemByType<T>(c.begin(), c.end());
}

/* Utility methods for MLIR-based IR */
namespace mlir::hil {

  // Tries to find a Graph by name; returns nullptr if unsuccessful.
  std::unique_ptr<mlir::hil::Graph> findGraph(mlir::hil::Graphs graphs,
                                              const std::string &name);
  
  // Tries to find a Chan by name; returns nullptr if unsuccessful.
  std::unique_ptr<mlir::hil::Chan> findChan(mlir::hil::Chans chans,
                                            const std::string &name);
  
  // Tries to find a Node by name; returns nullptr if unsuccessful.
  std::unique_ptr<mlir::hil::Node> findNode(mlir::hil::Nodes nodes,
                                            const std::string &name);

  // Tries to find a NodeType by name; returns nullptr if unsuccessful.
  std::unique_ptr<mlir::hil::NodeType> findNodetype(
      mlir::hil::NodeTypes nodeTypes,
      const std::string &name);

  // Tries to find an Inst by name; returns nullptr if unsuccessful.
  std::unique_ptr<mlir::hil::Inst> findInst(mlir::hil::Insts insts,
                                            const std::string &name);

  /// Returns parent model's name for the node.
  std::string getModelName(mlir::hil::Node &node);

  /// Returns parent model's name for the channel.
  std::string getModelName(mlir::hil::Chan &chan);

  /// Returns (if exist) named graph that belongs to the model.
  std::optional<Graph> getGraph(Model &model, const std::string &name);

  /* Graph-related methods. */
  std::vector<Node> getInputs(Graph &graph);
  std::vector<Node> getSinks(Graph &graph);
  std::vector<Chan> getChans(Graph &graph);
  mlir::Block::OpListType& getNodes(Graph &graph);

  /* Node-related methods. */
  std::vector<Chan> getInputs(Node &node);
  std::vector<Chan> getOutputs(Node &node);

  /* Type-checking methods. */
  bool isConst(Node &node);
  bool isDelay(Node &node);
  bool isDup(Node &node);
  bool isInstance(Node &node);
  bool isKernel(Node &node);
  bool isMerge(Node &node);
  bool isSink(Node &node);
  bool isSource(Node &node);
  bool isSplit(Node &node);

} // namespace mlir::hil