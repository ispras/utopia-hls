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

namespace mlir::hil {

  Graph* getGraph(Model &model, const std::string &name) {
    auto &model_ops = model.getBody()->getOperations();
    std::vector<Graph> graphs =
        find_elems_by_type<Graph>(model_ops.begin(), model_ops.end());

    Graph *main_g;

    for (size_t i = 0; i < graphs.size(); i++) {
      if (graphs[i].name() == name) {
        main_g = &graphs[i];
        break;
      }
    }
    assert(main_g && "Can't find \"main\" graph");
    return main_g;
  }

  std::vector<Chan*> getInputs(Node &node) {

    std::vector<Chan*> inChans;

    Graph graph = cast<Graph>(node->getParentOp()->getParentOp());

    std::vector<Chan*> chans = getChans(graph);

    for (auto arg : node.commandArguments()) {

      std::string in_chan_name = arg.cast<StringAttr>().getValue().str();

      for (size_t i = 0; i < chans.size(); i++) {
        if (chans[i]->varName().str() == in_chan_name) {
          inChans.push_back(chans[i]);
        }
      }
    }
    return inChans;
  }

  std::vector<Chan*> getOutputs(Node &node) {

    std::vector<Chan*> outChans;

    Graph graph = cast<Graph>(node->getParentOp()->getParentOp());

    std::vector<Chan*> chans = getChans(graph);

    for (auto res : node.commandResults()) {

      std::string out_chan_name = res.cast<StringAttr>().getValue().str();

      for (size_t i = 0; i < chans.size(); i++) {
        if (chans[i]->varName().str() == out_chan_name) {
          outChans.push_back(chans[i]);
        }
      }
    }
    return outChans;
  }

  std::vector<mlir::hil::Chan*> getChans(mlir::hil::Graph &graph) {

    auto graph_ops = graph.body().getOps();

    std::vector<mlir::hil::Chan> chans =
        find_elems_by_type<Chan>(graph_ops.begin(), graph_ops.end());
    std::vector<mlir::hil::Chan*> result;

    for (size_t i = 0; i < chans.size(); i++) {
      result.push_back(&chans[i]);
    }

    return result;
  }

  std::vector<mlir::hil::Node*> getNodes(mlir::hil::Graph &graph) {
    auto &graph_ops = graph.getBody()->getOperations();
    auto nodes_op = find_elem_by_type<Nodes>(graph_ops).value();
    std::vector<Node*> result;

    for (auto &nodes_block_op : nodes_op.getBody()->getOperations()) {
      auto node = cast<Node>(nodes_block_op);
      result.push_back(&node);
    }
    return result;
  }

  bool isDelay(mlir::hil::Node &node) {
    return node.nodeTypeName() == "delay";
  }

  bool isKernel(mlir::hil::Node &node) {
    return node.nodeTypeName() == "kernel";
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
} // namespace mlir::hil
