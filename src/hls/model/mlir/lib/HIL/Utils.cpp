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

#include "HIL/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "utils/string.h"

#include "llvm/Support/Casting.h"

#include <iostream>
#include <memory>

namespace mlir::hil {

std::unique_ptr<mlir::hil::Graph> findGraph(mlir::hil::Graphs graphs,
                                            const std::string &name) {
  std::unique_ptr<mlir::hil::Graph> result;
  graphs.walk([&](mlir::hil::Graph graphOp) {
      if (graphOp.name() == name) {
        result = std::make_unique<mlir::hil::Graph>(graphOp);
      }
  });
  return result;
}

std::unique_ptr<mlir::hil::Chan> findChan(mlir::hil::Chans chans,
                                          const std::string &name) {
  std::unique_ptr<mlir::hil::Chan> result;
  chans.walk([&](mlir::hil::Chan chanOp) {
      if (chanOp.varName() == name) {
        result = std::make_unique<mlir::hil::Chan>(chanOp);
      }
  });
  return result;
}

std::unique_ptr<mlir::hil::Node> findNode(mlir::hil::Nodes nodes,
                                          const std::string &name) {
  std::unique_ptr<mlir::hil::Node> result;
  nodes.walk([&](mlir::hil::Node nodeOp) {
      if (nodeOp.name() == name) {
        result = std::make_unique<mlir::hil::Node>(nodeOp);
      }
  });
  return result;
}

std::unique_ptr<mlir::hil::NodeType> findNodetype(
    mlir::hil::NodeTypes nodeTypes,
    const std::string &name) {
  std::unique_ptr<mlir::hil::NodeType> result;
  nodeTypes.walk([&](mlir::hil::NodeType nodeTypeOp) {
      if (nodeTypeOp.name() == name) {
        result = std::make_unique<mlir::hil::NodeType>(nodeTypeOp);
      }
  });
  return result;
}

std::unique_ptr<mlir::hil::Inst> findInst(mlir::hil::Insts insts,
                                          const std::string &name) {
  std::unique_ptr<mlir::hil::Inst> result;
  insts.walk([&](mlir::hil::Inst instOp) {
      if (instOp.name() == name) {
        result = std::make_unique<mlir::hil::Inst>(instOp);
      }
  });
  return result;
}

std::string getModelName(mlir::hil::Node &node) {
  auto model =
      mlir::cast<Model>(*node->getParentOp()->getParentOp()->getParentOp());
  return model.name().str();
}

std::string getModelName(mlir::hil::Chan &chan) {
  auto model =
      mlir::cast<Model>(*chan->getParentOp()->getParentOp()->getParentOp());
  return model.name().str();
}

std::optional<Graph> getGraph(Model &model, const std::string &name) {
  auto &modelOps = model.getBody()->getOperations();
  auto graphs = findElemsByType<Graph>(modelOps.begin(), modelOps.end());

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

    llvm::StringRef inChanName = arg.cast<StringAttr>().getValue();

    for (size_t i = 0; i < chans.size(); i++) {
      if (chans[i].varName() == inChanName) {
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

    llvm::StringRef outChanName = res.cast<StringAttr>().getValue();

    for (size_t i = 0; i < chans.size(); i++) {
      if (chans[i].varName() == outChanName) {
        outChans.push_back(chans[i]);
      }
    }
  }
  return outChans;
}

std::vector<Node> getInputs(mlir::hil::Graph &graph) {

  std::vector<Node> result;
  mlir::Block::OpListType &graphNodes = getNodes(graph);

  for (auto &graphNode : graphNodes) {

    auto node = cast<Node>(graphNode);
    if (isSource(node) || isConst(node)) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<Node> getSinks(mlir::hil::Graph &graph) {

  std::vector<Node> result;
  mlir::Block::OpListType &graphNodes = getNodes(graph);

  for (auto &graphNode : graphNodes) {

    auto node = cast<Node>(graphNode);
    if (isSink(node)) {
      result.push_back(node);
    }
  }
  return result;
}

std::vector<mlir::hil::Chan> getChans(mlir::hil::Graph &graph) {

  std::vector<mlir::hil::Chan> result;

  auto &graphOperations = graph.getBody()->getOperations();
  auto chansOp = findElemByType<Chans>(graphOperations).value();
  for (auto &chansOperation : chansOp.getBody()->getOperations()) {
    auto chan = cast<Chan>(chansOperation);
    result.push_back(chan);
  }

  return result;
}

mlir::Block::OpListType& getNodes(mlir::hil::Graph &graph) {
  auto &graphOperations = graph.getBody()->getOperations();
  auto nodesOp = findElemByType<Nodes>(graphOperations).value();
  return nodesOp.getBody()->getOperations();
}

bool isConst(mlir::hil::Node &node) {
  if (!getInputs(node).empty())
    return false;

  auto outputs = getOutputs(node);
  for (auto output : outputs) {
    auto bnd = output.nodeFromAttr();
    auto port = bnd.getPort();

    if (!port.getIsConst())
      return false;
  }

  return true;
}

bool isDelay(mlir::hil::Node &node) {
  return getInputs(node).size() == 1
      && getOutputs(node).size() == 1
      && eda::utils::starts_with(node.nodeTypeName().str(), "delay");
}

bool isDup(mlir::hil::Node &node) {
  return getInputs(node).size() == 1
      && eda::utils::starts_with(node.nodeTypeName().str(), "dup");
}

bool isMerge(mlir::hil::Node &node) {
  return getOutputs(node).size() == 1
      && eda::utils::starts_with(node.nodeTypeName().str(), "merge");
}

bool isSink(mlir::hil::Node &node) {
  return getOutputs(node).empty();
}

bool isSource(mlir::hil::Node &node) {
  return getInputs(node).empty() && !isConst(node);
}

bool isSplit(mlir::hil::Node &node) {
  return getInputs(node).size() == 1
      && eda::utils::starts_with(node.nodeTypeName().str(), "split");
}

bool isKernel(mlir::hil::Node &node) {
  return !isConst(node)
      && !isDelay(node)
      && !isDup(node)
      && !isMerge(node)
      && !isSink(node)
      && !isSource(node)
      && !isSplit(node);
}

} // namespace mlir::hil