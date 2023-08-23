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
// MLIR passes.
//
//===----------------------------------------------------------------------===//

#include "HIL/API.h"
#include "HIL/Combine.h"
#include "HIL/Dialect.h"
#include "HIL/Model.h"
#include "HIL/Ops.h"
#include "HIL/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>
#include <optional>
#include <typeinfo>

using ArrayAttr = mlir::ArrayAttr;
template<typename Type>
using ArrayRef = mlir::ArrayRef<Type>;
using Attribute = mlir::Attribute;
using BindingGraphAttr = mlir::hil::BindingGraphAttr;
using BindingAttr = mlir::hil::BindingAttr;
using Block = mlir::Block;
using ChansOp = mlir::hil::ChansOp;
using ChanOp = mlir::hil::ChanOp;
using ConsOp = mlir::hil::ConsOp;
using ConOp = mlir::hil::ConOp;
using DialectRegistry = mlir::DialectRegistry;
using FrozenRewritePatternSet = mlir::FrozenRewritePatternSet;
using GraphsOp = mlir::hil::GraphsOp;
using GraphOp = mlir::hil::GraphOp;
using GreedyRewriteConfig = mlir::GreedyRewriteConfig;
using LogicalResult = mlir::LogicalResult;
using MLIRContext = mlir::MLIRContext;
using ModelOp = mlir::hil::ModelOp;
using NodeTypesOp = mlir::hil::NodeTypesOp;
using NodeTypeOp = mlir::hil::NodeTypeOp;
using NodesOp = mlir::hil::NodesOp;
using NodeOp = mlir::hil::NodeOp;
using Operation = mlir::Operation;
template<typename Type = void>
using OperationPass = mlir::OperationPass<Type>;
template<typename Type>
using Option = mlir::Pass::Option<Type>;
using LLVMStringLiteral = llvm::StringLiteral;
using LLVMStringRef = llvm::StringRef;
template<typename Type>
using ListOption = mlir::Pass::ListOption<Type>;
using Pass = mlir::Pass;
using PatternRewriter = mlir::PatternRewriter;
using PortAttr = mlir::hil::PortAttr;
using RewritePattern = mlir::RewritePattern;
using RewritePatternSet = mlir::RewritePatternSet;
using StringAttr = mlir::StringAttr;
using TypeID = mlir::TypeID;

namespace {
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

class ChansRewritePass : public RewritePattern {
public:
  ChansRewritePass(MLIRContext *context)
      : RewritePattern(/*Root operation name to match against*/
                       ChansOp::getOperationName(),
                       /*benefit*/1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {

    // Detect channels container operation.
    auto chansOp = mlir::dyn_cast<ChansOp>(*operation);
    if (!chansOp) {
      return mlir::failure();
    }
    auto *context = chansOp.getContext();
    auto outerRegionOps = chansOp->getParentRegion()->getOps();

    // Find Nodes operation.
    auto nodesOp = findElemByType<NodesOp>(outerRegionOps).value();

    // Create inChanName->nodeName map.
    std::map<std::string, std::string> chanToSource;

    // Create outChanName->nodeName map.
    std::map<std::string, std::string> chanToTarget;

    // Iterate over Nodes' sub-operations, fill maps.
    for (auto &nodesBlockOp : nodesOp.getBodyBlock()->getOperations()) {
      auto nodeOp = mlir::cast<NodeOp>(nodesBlockOp);
      auto nodeName = nodeOp.getName().str();

      for (auto inChanOp : nodeOp.getCommandArguments()) {
        auto inChanName = inChanOp.cast<StringAttr>().getValue().str();
        chanToTarget[inChanName] = nodeName;
      }

      for (auto outChanOp : nodeOp.getCommandResults()) {
        auto outChanName = outChanOp.cast<StringAttr>().getValue().str();
        chanToSource[outChanName] = nodeName;
      }
    }

    // Copy channels into vector.
    std::vector<std::reference_wrapper<Operation>> vectorOperations;
    std::copy(chansOp.getBodyBlock()->getOperations().begin(),
              chansOp.getBodyBlock()->getOperations().end(),
              std::back_inserter(vectorOperations));

    // Iterate over channels.
    for (auto &chansBlockOpRef : vectorOperations) {

      auto &chansBlockOp = chansBlockOpRef.get();
      auto chanOp = mlir::cast<ChanOp>(chansBlockOp);
      auto chanName = chanOp.getVarName().str();

      std::optional<std::string> nodeFrom;
      auto nodeFromIt = chanToSource.find(chanName);

      if (nodeFromIt != chanToSource.end()) {
        nodeFrom = nodeFromIt->second;
      }

      std::optional<std::string> nodeTo;
      auto nodeToIt = chanToTarget.find(chanName);

      if (nodeToIt != chanToTarget.end()) {
        nodeTo = nodeToIt->second;
      }

      rewriter.setInsertionPoint(&chansBlockOp);
      rewriter.replaceOpWithNewOp<ChanOp>(
          &chansBlockOp, chanOp.getTypeName(), chanOp.getVarName(),
          BindingAttr{}.get(context, nodeFrom.value(),
              chanOp.getNodeFrom().getPort()),
          BindingAttr{}.get(context, nodeTo.value(),
              chanOp.getNodeTo().getPort()));
    }
    return mlir::success();
  }
};

class InsertDelayPass : public RewritePattern {
public:
  InsertDelayPass(MLIRContext *context, const std::string &chanName,
      const unsigned latency)
      : RewritePattern(ChanOp::getOperationName(), 1, context),
        chanName(chanName), latency(latency) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {

    // Detect requested channel.
    auto chanOp = mlir::dyn_cast<ChanOp>(*operation);
    if (!chanOp || chanOp.getVarName() != chanName) {
      return mlir::failure();
    }
    auto chansOp = mlir::cast<ChansOp>(*chanOp->getParentOp());
    auto modelOp = mlir::cast<ModelOp>(*chansOp->getParentOp()->getParentOp());
    auto &modelOperations = modelOp.getBodyBlock()->getOperations();
    auto nodeTypesOp = findElemByType<NodeTypesOp>(modelOperations).value();
    auto graphOp = findElemByType<GraphOp>(modelOperations).value();
    auto &graphOps = graphOp.getBodyBlock()->getOperations();
    auto nodesOp = findElemByType<NodesOp>(graphOps).value();
    auto nodesCount = nodesOp.getBodyBlock()->getOperations().size();

    auto chanType = chanOp.getTypeName();
    auto nodeFrom = chanOp.getNodeFromAttr();
    auto nodeTo = chanOp.getNodeToAttr();

    auto betweenTypeName =
        "delay_" + chanType.str() + "_" + std::to_string(latency);
    auto betweenName = betweenTypeName + "_" + std::to_string(nodesCount);

    // Check if we already added a delay.
    bool isDelayAdded = false;
    nodeTypesOp.walk([&](NodeTypeOp nodeTypeOp) {
      if (!isDelayAdded && nodeTypeOp.getName() == betweenTypeName) {
        isDelayAdded = true;
      }
    });
    if (isDelayAdded) {
      return mlir::failure();
    }

    auto *context = nodeTypesOp.getContext();
    // Add nodetype.
    auto inAttr = PortAttr::get(context, "in", chanType.str(),
       1.0, latency, false, 0);
    auto outAttr = PortAttr::get(context, "out", chanType.str(),
       1.0, latency, false, 0);
    std::array<Attribute, 1> inAttrs{inAttr};
    std::array<Attribute, 1> outAttrs{outAttr};
    rewriter.setInsertionPointToEnd(nodeTypesOp.getBodyBlock());
    rewriter.create<NodeTypeOp>(
        nodeTypesOp.getLoc(), StringAttr{}.get(context, betweenTypeName),
        ArrayAttr::get(context, inAttrs), ArrayAttr::get(context, outAttrs));
    auto newChanName = betweenName + "_out";
    // Add a splitting node.
    rewriter.setInsertionPointToEnd(nodesOp.getBodyBlock());
    std::array<Attribute, 1> inChans{chanOp.getVarNameAttr()};
    std::array<Attribute, 1> outChans{
        StringAttr{}.get(context, newChanName)};
    rewriter.create<NodeOp>(
        nodesOp.getLoc(), StringAttr{}.get(context, betweenTypeName),
        StringAttr{}.get(context, betweenName),
        ArrayAttr::get(context, inChans), ArrayAttr::get(context, outChans));
    // Split the channel with the node.
    rewriter.setInsertionPointToEnd(chansOp.getBodyBlock());
    rewriter.create<ChanOp>(chansOp.getLoc(), chanOp.getTypeName(), newChanName,
        BindingAttr{}.get(context, betweenName, outAttr), nodeTo);
    rewriter.replaceOpWithNewOp<ChanOp>(chanOp, chanOp.getTypeName(),
        chanOp.getVarName(), nodeFrom,
        BindingAttr{}.get(context, betweenName, inAttr));
    // Rename target node's input channel.
    nodesOp.walk([&](NodeOp nodeOp) {
      if (nodeOp.getName() == nodeTo.getNodeName()) {
        auto &&args = nodeOp.getCommandArguments();
        std::vector<Attribute> inChans{args.begin(), args.end()};
        for (auto &inChanName : inChans) {
          if (inChanName.cast<StringAttr>().getValue() == chanName) {
            inChanName = StringAttr{}.get(context, newChanName);
            break;
          }
        }
        rewriter.setInsertionPoint(nodeOp);
        rewriter.replaceOpWithNewOp<NodeOp>(
            nodeOp, nodeOp.getNodeTypeNameAttr(), nodeOp.getNameAttr(),
            ArrayAttr::get(context, inChans), nodeOp.getCommandResultsAttr());
      }
    });
    return mlir::success();
  }

private:
  const std::string chanName;
  const unsigned latency;
};

class UnfoldInstancePass : public RewritePattern {
public:
  UnfoldInstancePass(MLIRContext *context,
                     const std::string &instanceName,
                     const std::string &instanceGraphName,
                     const std::string &mainGraphName)
      : RewritePattern(NodeOp::getOperationName(), 1, context),
        instanceName(instanceName), 
        instanceGraphName(instanceGraphName),
        mainGraphName(mainGraphName) {}

  LogicalResult matchAndRewrite(Operation *operation,
                                PatternRewriter &rewriter) const override {
    // MATCH PART.
    //--------------------------------------------------------------------------
    // Get the requested instance.
    auto nodeInstOp = mlir::dyn_cast<NodeOp>(operation);
    if (!nodeInstOp || nodeInstOp.getName() != instanceName) {
      return mlir::failure();
    }
    //--------------------------------------------------------------------------
    // REWRITE PART.
    //--------------------------------------------------------------------------
  
    // Get the main Graph and the instance Graph.
    auto modelOp = nodeInstOp->getParentOfType<ModelOp>();
    auto &modelOperations = modelOp.getBodyBlock()->getOperations();
    auto graphsOp = findElemByType<GraphsOp>(modelOperations).value();
    auto graphInstOp = findGraph(graphsOp, instanceGraphName);
    auto graphMainOp = findGraph(graphsOp, mainGraphName);

    // Get the main and the instance graph operations.
    auto &graphInstOperations = graphInstOp->getBodyBlock()->getOperations();
    auto &graphMainOperations = graphMainOp->getBodyBlock()->getOperations();

    // Get Nodes from the main Graph and the instance Graph.
    auto nodesInstOp = findElemByType<NodesOp>(graphInstOperations).value();
    auto nodesMainOp = findElemByType<NodesOp>(graphMainOperations).value();

    auto *context = nodesMainOp.getContext();
    // Clone Nodes from the instance graph to the main Graph.
    // NOTE: There may be multiple instances so name change is needed.
    rewriter.setInsertionPointToEnd(nodesMainOp.getBodyBlock());
    nodesInstOp.walk([&](NodeOp nodeOp) {
      // Changing the name of the clone Node and 
      // the names of the inputs...
      auto &&args = nodeOp.getCommandArguments();
      std::vector<Attribute> newInChanNames;
      for (const auto arg : args) {
        auto newInChanName = arg.cast<StringAttr>().str() + "_" 
                                                          + instanceName;
        newInChanNames.push_back(StringAttr{}.get(context, newInChanName));
      }
      // ...and the names of the outputs. 
      auto &&ress = nodeOp.getCommandResults();
      std::vector<Attribute> newOutChanNames;
      for (const auto res : ress) {
        auto newOutChanName = res.cast<StringAttr>().str() + "_" 
                                                           + instanceName;
        newOutChanNames.push_back(StringAttr{}.get(context, newOutChanName));
      }
      // Creating a new Node with the modified name.
      auto newNodeName = nodeOp.getNameAttr().str() + "_" + instanceName;
      /// TODO: Try to find a more convenient way to create a container.
      auto newNodeOp = rewriter.create<NodeOp>(nodesMainOp.getLoc(),
          nodeOp.getNodeTypeNameAttr(), newNodeName,
          ArrayAttr::get(context, newInChanNames),
          ArrayAttr::get(context, newOutChanNames));
      Block *body = new Block();
      newNodeOp.getRegion().push_back(body);
    });

    // Get Chans from the main Graph and the instance Graph.
    auto chansInstOp = findElemByType<ChansOp>(graphInstOperations).value();
    auto chansMainOp = findElemByType<ChansOp>(graphMainOperations).value();

    // Clone Chans from the instance graph to the main Graph.
    // NOTE: There may be multiple instances so name change is needed.
    rewriter.setInsertionPointToEnd(chansMainOp.getBodyBlock());
    chansInstOp.walk([&](ChanOp chanOp) {
      // Creating a new Chan with the modified names.
      auto newChanName = chanOp.getVarName().str() + "_" + instanceName;
      auto newNodeFromName = chanOp.getNodeFromAttr().getNodeName().str() + "_"
                           + instanceName;
      auto newNodeToName = chanOp.getNodeToAttr().getNodeName().str() + "_"
                         + instanceName;
      rewriter.create<ChanOp>(chansMainOp.getLoc(),
                              chanOp.getTypeName(),
                              newChanName,
                              BindingAttr{}.get(context, newNodeFromName,
                                                chanOp.getNodeFrom().getPort()),
                              BindingAttr{}.get(context, newNodeToName,
                                                chanOp.getNodeTo().getPort()));
    });

    // Get main Graph connections.
    auto &nodeInstOpOperations = nodeInstOp.getBodyBlock()->getOperations();
    auto consInstOp = findElemByType<ConsOp>(nodeInstOpOperations).value();

    // Modify the Chans in the main Graph and in the instance Graph.
    std::map<std::string, std::vector<Attribute>> nodeToNewInChanNames;
    std::map<std::string, std::vector<Attribute>> nodeToNewOutChanNames;
    consInstOp.walk([&](ConOp conOp) {
      auto sourceBndGraph = conOp.getNodeFromAttr();
      auto targetBndGraph = conOp.getNodeToAttr();
      auto dirTypeName = conOp.getDirTypeName();
      // Connection to input.
      if (dirTypeName == "IN") {
        auto sourceChanName = sourceBndGraph.getChanName().str();
        auto targetChanNameAfter = targetBndGraph.getChanName().str() + "_" 
                                 + instanceName;
        
        auto sourceChanOp = findChan(chansMainOp,
                                     sourceChanName);
        auto targetChanAfterOp = findChan(chansMainOp,
                                          targetChanNameAfter);

        rewriter.setInsertionPointToEnd(chansMainOp.getBodyBlock());
        rewriter.replaceOpWithNewOp<ChanOp>(
           *sourceChanOp,
            sourceChanOp->getTypeName(),
            sourceChanOp->getVarName(),
            sourceChanOp->getNodeFrom(),
            BindingAttr{}.get(context, 
                targetChanAfterOp->getNodeToAttr().getNodeName(),
                targetChanAfterOp->getNodeTo().getPort()));
        // Save the name of the input chan to modify it in the node later.
        auto nodeToName = 
            targetChanAfterOp->getNodeToAttr().getNodeName().str();
        auto iterator = nodeToNewInChanNames.find(nodeToName);
        std::vector<Attribute> newVectorAttr;
        if (iterator == nodeToNewInChanNames.end()) {
          nodeToNewInChanNames.emplace(nodeToName,
                                       newVectorAttr);
        }
        iterator = nodeToNewInChanNames.find(nodeToName);
        auto newChanNameStr = sourceChanOp->getVarName().str();
        auto newChanNameAttr = StringAttr{}.get(context, newChanNameStr);
        iterator->second.push_back(newChanNameAttr);
        // Delete the 'source' Node.
        auto sourceNodeNameAfter = 
            targetChanAfterOp->getNodeFromAttr().getNodeName().str();
        auto sourceNode = findNode(nodesMainOp, sourceNodeNameAfter);
        rewriter.eraseOp(*sourceNode);
        // Delete the Chan connecting the 'source' Node.
        rewriter.eraseOp(*targetChanAfterOp);
      }

      // Connection to output.
      if (dirTypeName == "OUT") {
        auto targetChanName = targetBndGraph.getChanName().str();
        auto sourceChanNameAfter = sourceBndGraph.getChanName().str() + "_" 
                                 + instanceName;
      
        auto sourceChanAfterOp = findChan(chansMainOp,
                                          sourceChanNameAfter);
        auto targetChanOp = findChan(chansMainOp,
                                     targetChanName);
        rewriter.setInsertionPointToEnd(chansMainOp.getBodyBlock());
        rewriter.replaceOpWithNewOp<ChanOp>(
           *targetChanOp,
            targetChanOp->getTypeName(),
            targetChanOp->getVarName(),
            BindingAttr{}.get(context,
                sourceChanAfterOp->getNodeFrom().getNodeName(),
                sourceChanAfterOp->getNodeFrom().getPort()),
            targetChanOp->getNodeTo());
        // Save the name of the output chan to modify it in the node later.
        auto nodeFromName = 
            sourceChanAfterOp->getNodeFromAttr().getNodeName().str();
        auto iterator = nodeToNewOutChanNames.find(nodeFromName);
        std::vector<Attribute> newVectorAttr;
        if (iterator == nodeToNewOutChanNames.end()) {
          nodeToNewOutChanNames.emplace(nodeFromName,
                                        newVectorAttr);
        }
        iterator = nodeToNewOutChanNames.find(nodeFromName);
        auto newChanNameStr = targetChanOp->getVarName().str();
        auto newChanNameAttr = StringAttr{}.get(context, newChanNameStr);
        iterator->second.push_back(newChanNameAttr);
        // Delete the 'sink' Node.
        auto sinkNodeNameAfter = 
            sourceChanAfterOp->getNodeToAttr().getNodeName().str();
        auto sinkNode = findNode(nodesMainOp, sinkNodeNameAfter);
        rewriter.eraseOp(*sinkNode);
        // Delete the Chan connecting the 'sink' Node.
        rewriter.eraseOp(*sourceChanAfterOp);
      }
      // Delete connection.
      rewriter.eraseOp(conOp);
    });
    rewriter.setInsertionPointToEnd(nodesMainOp.getBodyBlock());
    for (const auto &pair : nodeToNewInChanNames) {
      auto nodeOp = findNode(nodesMainOp, pair.first);
      auto inChans = nodeToNewInChanNames.find(pair.first)->second;
      auto outChans = nodeToNewOutChanNames.find(pair.first)->second;
      /// TODO: Try to find a more convenient way to create a container.
      Block *block = new Block();
      auto newNodeOp = rewriter.replaceOpWithNewOp<NodeOp>(*nodeOp,
          nodeOp->getNodeTypeNameAttr(), nodeOp->getName(),
          ArrayAttr::get(context, inChans), ArrayAttr::get(context, outChans));
      newNodeOp.getRegion().push_back(block);
    }
    // Delete connections container.
    rewriter.eraseOp(consInstOp);
    // Delete the instance node in the main Graph.
    rewriter.eraseOp(nodeInstOp);
    return mlir::success();
  }

private:
  const std::string instanceName;
  const std::string instanceGraphName;
  const std::string mainGraphName;
};

} // namespace

namespace {
template <typename DerivedT>
class CanonicalizerBase : public OperationPass<ModuleOp> {
public:
  using Base = CanonicalizerBase;

  CanonicalizerBase()
      : OperationPass<ModuleOp>(TypeID::get<DerivedT>()) {}
  CanonicalizerBase(const CanonicalizerBase &other)
      : OperationPass<ModuleOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr LLVMStringLiteral getArgumentName() {
    return LLVMStringLiteral("canonicalize");
  }
  LLVMStringRef getArgument() const override { return "canonicalize"; }

  LLVMStringRef getDescription() const override {
    return "Canonicalize operations";
  }

  /// Returns the derived pass name.
  static constexpr LLVMStringLiteral getPassName() {
    return LLVMStringLiteral("Canonicalizer");
  }
  LLVMStringRef getName() const override { return "Canonicalizer"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Pass *pass) {
    return pass->getTypeID() == TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(DialectRegistry &registry) const override {}
};

class GraphCanonicalizer : public CanonicalizerBase<GraphCanonicalizer> {
public:
  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = false;
    config.maxIterations = 1;
    RewritePatternSet patterns(&getContext());
    patterns.add<UnfoldInstancePass>(&getContext(),
                                     "VectorSum1",
                                     "VectorSum",
                                     "InstanceTest");

    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns),
                                                  config)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::transforms {
void runPass(MLIRModule &mlirModule, RewritePattern &&pass) {
  auto *context = mlirModule.getContext();
  SimpleRewriter rewriter(context);
  mlirModule.getRoot()->walk(
      [&](Operation *operation) { (void)pass.matchAndRewrite(operation,
                                                             rewriter); });
}

std::function<void(MLIRModule &)> ChanAddSourceTarget() {
  return [=](MLIRModule &mlirModule) {
    auto *context = mlirModule.getContext();
    ChansRewritePass pass{context};
    runPass(mlirModule, std::move(pass));
  };
}

std::function<void(MLIRModule &)> InsertDelay(const std::string &chanName,
                                              const unsigned latency) {
  return [=](MLIRModule &mlirModule) {
    auto *context = mlirModule.getContext();
    InsertDelayPass pass{context, chanName, latency};
    runPass(mlirModule, std::move(pass));
  };
}

std::function<void(MLIRModule &)> UnfoldInstance(
  const std::string &instanceName, const std::string &instanceGraphName,
  const std::string &mainGraphName) {
  return [=](MLIRModule &mlirModule) {
    auto *context = mlirModule.getContext();
    UnfoldInstancePass pass{context,
                            instanceName,
                            instanceGraphName,
                            mainGraphName};
    runPass(mlirModule, std::move(pass));
  };
}

} // namespace mlir::transforms

std::unique_ptr<Pass> createGraphRewritePass() {
  return std::make_unique<GraphCanonicalizer>();
}