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

using LLVMStringRef = llvm::StringRef;
using OpListType = mlir::Block::OpListType;

namespace mlir::hil {

std::unique_ptr<GraphOp> findGraph(GraphsOp graphsOp, const std::string &name) {
  std::unique_ptr<GraphOp> result;
  graphsOp.walk([&](GraphOp graphOp) {
      if (graphOp.getName() == name) {
        result = std::make_unique<GraphOp>(graphOp);
        WalkResult::interrupt();
      }
  });
  return result;
}

std::unique_ptr<ChanOp> findChan(ChansOp chansOp, const std::string &name) {
  std::unique_ptr<ChanOp> result;
  chansOp.walk([&](ChanOp chanOp) {
      if (chanOp.getVarName() == name) {
        result = std::make_unique<ChanOp>(chanOp);
        WalkResult::interrupt();
      }
  });
  return result;
}

std::unique_ptr<NodeOp> findNode(NodesOp nodesOp, const std::string &name) {
  std::unique_ptr<NodeOp> result;
  nodesOp.walk([&](NodeOp nodeOp) {
      if (nodeOp.getName() == name) {
        result = std::make_unique<NodeOp>(nodeOp);
        WalkResult::interrupt();
      }
  });
  return result;
}

std::unique_ptr<NodeTypeOp> findNodetype(NodeTypesOp nodeTypesOp,
                                         const std::string &name) {
  std::unique_ptr<NodeTypeOp> result;
  nodeTypesOp.walk([&](NodeTypeOp nodeTypeOp) {
      if (nodeTypeOp.getName() == name) {
        result = std::make_unique<NodeTypeOp>(nodeTypeOp);
        WalkResult::interrupt();
      }
  });
  return result;
}

std::string getModelName(NodeOp &nodeOp) {
  auto model = nodeOp->getParentOfType<ModelOp>();
  return model.getName().str();
}

std::string getModelName(ChanOp &chanOp) {
  auto model = chanOp->getParentOfType<ModelOp>();
  return model.getName().str();
}

std::optional<GraphOp> getGraphOp(ModelOp &modelOp, const std::string &name) {
  auto &modelOps = modelOp.getBodyBlock()->getOperations();
  auto graphs = findElemsByType<GraphOp>(modelOps.begin(), modelOps.end());
  for (auto &graph : graphs) {
    if (graph.getName() == name) {
      return graph;
    }
  }
  return std::nullopt;
}

std::vector<ChanOp> getInputs(NodeOp &nodeOp) {
  std::vector<ChanOp> inChans;
  auto graphOp = nodeOp->getParentOfType<GraphOp>();
  std::vector<ChanOp> chanOpVector = getChans(graphOp);
  for (auto arg : nodeOp.getCommandArguments()) {
    LLVMStringRef inChanName = arg.cast<StringAttr>().getValue();
    for (auto &chanOp : chanOpVector) {
      if (chanOp.getVarName() == inChanName) {
        inChans.push_back(chanOp);
      }
    }
  }
  return inChans;
}

std::vector<ChanOp> getOutputs(NodeOp &nodeOp) {
  std::vector<ChanOp> outChans;
  GraphOp graphOp = cast<GraphOp>(nodeOp->getParentOp()->getParentOp());
  std::vector<ChanOp> chanOpVector = getChans(graphOp);
  for (auto res : nodeOp.getCommandResults()) {
    LLVMStringRef outChanName = res.cast<StringAttr>().getValue();
    for (auto &chanOp : chanOpVector) {
      if (chanOp.getVarName() == outChanName) {
        outChans.push_back(chanOp);
      }
    }
  }
  return outChans;
}

std::vector<NodeOp> getSourcesAndConsts(GraphOp &graphOp) {
  std::vector<NodeOp> result;
  OpListType &graphNodes = getNodes(graphOp);
  for (auto &graphNode : graphNodes) {
    auto nodeOp = cast<NodeOp>(graphNode);
    if (isSource(nodeOp) || isConst(nodeOp)) {
      result.push_back(nodeOp);
    }
  }
  return result;
}

std::vector<NodeOp> getSinks(GraphOp &graphOp) {
  std::vector<NodeOp> result;
  OpListType &graphNodes = getNodes(graphOp);
  for (auto &graphNode : graphNodes) {
    auto nodeOp = cast<NodeOp>(graphNode);
    if (isSink(nodeOp)) {
      result.push_back(nodeOp);
    }
  }
  return result;
}

std::vector<ChanOp> getChans(GraphOp &graphOp) {
  std::vector<ChanOp> result;
  auto &graphOperations = graphOp.getBodyBlock()->getOperations();
  auto chansOp = findElemByType<ChansOp>(graphOperations).value();
  for (auto &chansOperation : chansOp.getBodyBlock()->getOperations()) {
    auto chanOp = cast<ChanOp>(chansOperation);
    result.push_back(chanOp);
  }
  return result;
}

OpListType &getNodes(GraphOp &graphOp) {
  auto &graphOperations = graphOp.getBodyBlock()->getOperations();
  auto nodesOp = findElemByType<NodesOp>(graphOperations).value();
  return nodesOp.getBodyBlock()->getOperations();
}

bool isConst(NodeOp &nodeOp) {
  if (!getInputs(nodeOp).empty())
    return false;

  auto outputs = getOutputs(nodeOp);
  for (auto &output : outputs) {
    auto binding = output.getNodeFrom();
    auto portAttr = binding.getPort();

    if (!portAttr.getIsConst())
      return false;
  }

  return true;
}

bool isDelay(NodeOp &nodeOp) {
  return getInputs(nodeOp).size() == 1
      && getOutputs(nodeOp).size() == 1
      && eda::utils::starts_with(nodeOp.getNodeTypeName().str(), "delay");
}

bool isDup(NodeOp &nodeOp) {
  return getInputs(nodeOp).size() == 1
      && eda::utils::starts_with(nodeOp.getNodeTypeName().str(), "dup");
}

bool isMerge(NodeOp &nodeOp) {
  return getOutputs(nodeOp).size() == 1
      && eda::utils::starts_with(nodeOp.getNodeTypeName().str(), "merge");
}

bool isSink(NodeOp &nodeOp) {
  return getOutputs(nodeOp).empty();
}

bool isSource(NodeOp &nodeOp) {
  return getInputs(nodeOp).empty() && !isConst(nodeOp);
}

bool isSplit(NodeOp &nodeOp) {
  return getInputs(nodeOp).size() == 1
      && eda::utils::starts_with(nodeOp.getNodeTypeName().str(), "split");
}

bool isInstance(NodeOp &nodeOp) {
  return eda::utils::starts_with(nodeOp.getNodeTypeName().str(), "INSTANCE");
}

bool isKernel(NodeOp &nodeOp) {
  return !isConst(nodeOp)
      && !isDelay(nodeOp)
      && !isDup(nodeOp)
      && !isMerge(nodeOp)
      && !isSink(nodeOp)
      && !isSource(nodeOp)
      && !isSplit(nodeOp);
}

} // namespace mlir::hil