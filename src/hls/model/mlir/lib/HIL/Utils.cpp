//===- HILOps.cpp - MLIR utils      ---------------*- C++ -*---------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIL/Utils.h"

#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <iostream>

namespace mlir::hil {

  std::optional<Graph> getGraph(Model &model, const std::string &name) {
    auto &model_ops = model.getBody()->getOperations();
    auto graphs = find_elems_by_type<Graph>(model_ops.begin(), model_ops.end());

    for (size_t i = 0; i < graphs.size(); i++) {
      if (graphs[i].name() == name) {
        return graphs[i];
      }
    }
    return std::nullopt;
  }

  std::vector<Chan> getInputs(Node &node) {

    std::vector<Chan> inChans;

    Graph graph = cast<Graph>(node->getParentOp()->getParentOp());

    std::vector<Chan> chans = getChans(graph);

    for (auto arg : node.commandArguments()) {

      llvm::StringRef in_chan_name = arg.cast<StringAttr>().getValue();

      for (size_t i = 0; i < chans.size(); i++) {
        if (chans[i].varName() == in_chan_name) {
          inChans.push_back(chans[i]);
        }
      }
    }
    return inChans;
  }

  std::vector<Chan> getOutputs(Node &node) {

    std::vector<Chan> outChans;

    Graph graph = cast<Graph>(node->getParentOp()->getParentOp());

    std::vector<Chan> chans = getChans(graph);

    for (auto res : node.commandResults()) {

      llvm::StringRef out_chan_name = res.cast<StringAttr>().getValue();

      for (size_t i = 0; i < chans.size(); i++) {
        if (chans[i].varName() == out_chan_name) {
          outChans.push_back(chans[i]);
        }
      }
    }
    return outChans;
  }

  std::vector<Node> getSources(mlir::hil::Graph &graph) {

  std::vector<Node> result;
  mlir::Block::OpListType &graphNodes = getNodes(graph);

  for (auto &gNode : graphNodes) {

    auto node = cast<Node>(gNode);
    if (isSource(node)) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<Node> getSinks(mlir::hil::Graph &graph) {

  std::vector<Node> result;
  mlir::Block::OpListType &graphNodes = getNodes(graph);

  for (auto &gNode : graphNodes) {

    auto node = cast<Node>(gNode);
    if (isSink(node)) {
      result.push_back(node);
    }
  }
  return result;
}

  std::vector<mlir::hil::Chan> getChans(mlir::hil::Graph &graph) {

    std::vector<mlir::hil::Chan> result;

    auto &g_ops = graph.getBody()->getOperations();
    auto chans_ops = find_elem_by_type<Chans>(g_ops).value();
    for (auto &chans_op : chans_ops.getBody()->getOperations()) {
      auto chan = cast<Chan>(chans_op);
      result.push_back(chan);
    }

    return result;
  }

  mlir::Block::OpListType& getNodes(mlir::hil::Graph &graph) {
    auto &graph_ops = graph.getBody()->getOperations();
    auto nodes_op = find_elem_by_type<Nodes>(graph_ops).value();
    return nodes_op.getBody()->getOperations();
  }

  bool isDelay(mlir::hil::Node &node) {
    return node.nodeTypeName() == "delay";
  }

  bool isDup(mlir::hil::Node &node) {
    return node.nodeTypeName() == "dup";
  }

  bool isMerge(mlir::hil::Node &node) {
    return node.nodeTypeName() == "merge";
  }

  bool isSink(mlir::hil::Node &node) {
    return node.nodeTypeName() == "sink";
  }

  bool isSource(mlir::hil::Node &node) {
    return node.nodeTypeName() == "source";
  }

  bool isSplit(mlir::hil::Node &node) {
    return node.nodeTypeName() == "split";
  }

  bool isKernel(mlir::hil::Node &node) {
    return !isDelay(node)
        && !isDup(node)
        && !isMerge(node)
        && !isSink(node)
        && !isSource(node)
        && !isSplit(node);
  }
} // namespace mlir::hil
