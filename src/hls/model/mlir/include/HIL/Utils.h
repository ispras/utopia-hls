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

#include "mlir/IR/BuiltinAttributes.h"

#include "HIL/Ops.h"
#include "mlir/Support/LLVM.h"

#include <optional>
#include <vector>

using BindingGraphAttr = mlir::hil::BindingGraphAttr;
using BindingAttr = mlir::hil::BindingAttr;
using ChansOp = mlir::hil::ChansOp;
using ChanOp = mlir::hil::ChanOp;
using ConsOp = mlir::hil::ConsOp;
using ConOp = mlir::hil::ConOp;
// using FModuleOp = circt::firrtl::FModuleOp;
using GraphsOp = mlir::hil::GraphsOp;
using GraphOp = mlir::hil::GraphOp;
// using InstanceOp = circt::firrtl::InstanceOp;
using ModelOp = mlir::hil::ModelOp;
using NodeTypesOp = mlir::hil::NodeTypesOp;
using NodeTypeOp = mlir::hil::NodeTypeOp;
using NodesOp = mlir::hil::NodesOp;
using NodeOp = mlir::hil::NodeOp;
using PortAttr = mlir::hil::PortAttr;

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

template <typename T, typename Container>
std::vector<T> findElemsByType(Container &&c) {
  return findElemsByType<T>(c.begin(), c.end());
}

/* Utility methods for MLIR-based IR */
namespace mlir::hil {
  // // Tries to find a FModule by name; returns nullptr if unsuccessful.
  // std::unique_ptr<FModuleOp> findFModule(Operation *op,
  //                                        const std::string &name);

  // // Tries to find an Instance by name; returns nullptr if unsuccessful.
  // std::unique_ptr<InstanceOp> findInstance(Operation *op,
  //                                          const std::string &name);
  
  // Tries to find a Chan by name; returns nullptr if unsuccessful.
  std::unique_ptr<ChanOp> findChan(ChansOp chans, const std::string &name);

  // Tries to find a Graph by name; returns nullptr if unsuccessful.
  std::unique_ptr<GraphOp> findGraph(GraphsOp graphs, const std::string &name);
  
  // Tries to find a Chan by name; returns nullptr if unsuccessful.
  std::unique_ptr<ChanOp> findChan(ChansOp chans, const std::string &name);
  
  // Tries to find a Node by name; returns nullptr if unsuccessful.
  std::unique_ptr<NodeOp> findNode(NodesOp nodes, const std::string &name);

  // Tries to find a NodeType by name; returns nullptr if unsuccessful.
  std::unique_ptr<NodeTypeOp> findNodeType(NodeTypesOp nodeTypes,
                                           const std::string &name);

  /// Returns parent model's name for the node.
  std::string getModelName(NodeOp &nodeOp);

  /// Returns parent model's name for the channel.
  std::string getModelName(ChanOp &chanOp);

  /// Returns (if exist) named graph that belongs to the model.
  std::optional<GraphOp> getGraphOp(ModelOp &modelOp, const std::string &name);

  /* Graph-related methods. */
  std::vector<NodeOp> getSourcesAndConsts(GraphOp &graphOp);
  std::vector<NodeOp> getSinks(GraphOp &graphOp);
  std::vector<ChanOp> getChans(GraphOp &graphOp);
  mlir::Block::OpListType& getNodes(GraphOp &graphOp);

  /* Node-related methods. */
  std::vector<ChanOp> getInputs(NodeOp &nodeOp);
  std::vector<ChanOp> getOutputs(NodeOp &nodeOp);

  /* Type-checking methods. */
  bool isConst(NodeOp &nodeOp);
  bool isDelay(NodeOp &nodeOp);
  bool isDup(NodeOp &nodeOp);
  bool isInstance(NodeOp &nodeOp);
  bool isKernel(NodeOp &nodeOp);
  bool isMerge(NodeOp &nodeOp);
  bool isSink(NodeOp &nodeOp);
  bool isSource(NodeOp &nodeOp);
  bool isSplit(NodeOp &nodeOp);

  bool isInstance(NodeTypeOp &nodeTypeOp);
  bool isSink(NodeTypeOp &nodeTypeOp);
  bool isSource(NodeTypeOp &nodeTypeOp);

} // namespace mlir::hil