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

#include "HIL/Combine.h"

#include "HIL/API.h"
#include "HIL/Dialect.h"
#include "HIL/Model.h"
#include "HIL/Ops.h"
#include "HIL/Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>
#include <optional>

#include <typeinfo>

using namespace mlir;
using namespace mlir::hil;

namespace {
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

class ChansRewritePass : public RewritePattern {
public:
  ChansRewritePass(MLIRContext *context)
      : RewritePattern(/*Root operation name to match against*/
                       mlir::hil::Chans::getOperationName(),
                       /*benefit*/1,
                       context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    // Detect channels container operation.
    auto chansOp = dyn_cast<Chans>(*op);
    if (!chansOp) {
      return failure();
    }
    auto *context = chansOp.getContext();
    auto outerRegionOps = chansOp->getParentRegion()->getOps();

    // Find Nodes operation.
    auto nodesOp = findElemByType<Nodes>(outerRegionOps).value();

    // Create inChanName->nodeName map.
    std::map<std::string, std::string> chanToSource;

    // Create outChanName->nodeName map.
    std::map<std::string, std::string> chanToTarget;

    // Iterate over Nodes' sub-operations, fill maps.
    for (auto &nodesBlockOp : nodesOp.getBody()->getOperations()) {
      auto nodeOp = cast<Node>(nodesBlockOp);
      auto nodeName = nodeOp.name().str();

      for (auto inChanOp : nodeOp.commandArguments()) {
        auto inChanName = inChanOp.cast<StringAttr>().getValue().str();
        chanToTarget[inChanName] = nodeName;
      }

      for (auto outChanOp : nodeOp.commandResults()) {
        auto outChanName = outChanOp.cast<StringAttr>().getValue().str();
        chanToSource[outChanName] = nodeName;
      }
    }

    // Copy channels into vector.
    std::vector<std::reference_wrapper<Operation>> vectorOperations;
    std::copy(chansOp.getBody()->getOperations().begin(),
              chansOp.getBody()->getOperations().end(),
              std::back_inserter(vectorOperations));

    // Iterate over channels.
    for (auto &chansBlockOpRef : vectorOperations) {

      auto &chansBlockOp = chansBlockOpRef.get();
      auto chanOp = cast<Chan>(chansBlockOp);
      auto chanName = chanOp.varName().str();

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
      rewriter.replaceOpWithNewOp<Chan>(
          &chansBlockOp, chanOp.typeName(), chanOp.varName(),
          BindingAttr{}.get(context, nodeFrom.value(),
              chanOp.nodeFrom().getPort()),
          BindingAttr{}.get(context, nodeTo.value(),
              chanOp.nodeTo().getPort()));
    }
    return success();
  }
};

class InsertDelayPass : public RewritePattern {
public:
  InsertDelayPass(MLIRContext *context, const std::string &chanName,
      const unsigned latency)
      : RewritePattern(mlir::hil::Chan::getOperationName(), 1, context),
        chanName(chanName), latency(latency) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    // Detect requested channel.
    auto chanOp = dyn_cast<Chan>(*op);
    if (!chanOp || chanOp.varName() != chanName) {
      return failure();
    }
    auto chansOp = cast<Chans>(*chanOp->getParentOp());
    auto modelOp = cast<mlir::hil::Model>(*chansOp->getParentOp()->getParentOp());
    auto &modelOperations = modelOp.getBody()->getOperations();
    auto nodeTypesOp = findElemByType<NodeTypes>(modelOperations).value();
    auto graphOp = findElemByType<Graph>(modelOperations).value();
    auto &graphOps = graphOp.getBody()->getOperations();
    auto nodesOp = findElemByType<Nodes>(graphOps).value();
    auto nodesCount = nodesOp.getBody()->getOperations().size();

    auto chanType = chanOp.typeName();
    auto nodeFrom = chanOp.nodeFromAttr();
    auto nodeTo = chanOp.nodeToAttr();

    auto betweenTypeName =
        "delay_" + chanType.str() + "_" + std::to_string(latency);
    auto betweenName = betweenTypeName + "_" + std::to_string(nodesCount);

    // Check if we already added a delay.
    bool isDelayAdded = false;
    nodeTypesOp.walk([&](NodeType op) {
      if (!isDelayAdded && op.name() == betweenTypeName) {
        isDelayAdded = true;
      }
    });
    if (isDelayAdded) {
      return failure();
    }

    auto *context = nodeTypesOp.getContext();
    // Add nodetype.
    auto inAttr = PortAttr::get(context, "in", chanType.str(),
       1.0, latency, false, 0);
    auto outAttr = PortAttr::get(context, "out", chanType.str(),
       1.0, latency, false, 0);
    std::array<Attribute, 1> inAttrs{inAttr};
    std::array<Attribute, 1> outAttrs{outAttr};
    rewriter.setInsertionPointToEnd(nodeTypesOp.getBody());
    rewriter.create<NodeType>(
        nodeTypesOp.getLoc(), StringAttr{}.get(context, betweenTypeName),
        ArrayAttr::get(context, inAttrs), ArrayAttr::get(context, outAttrs));
    auto newChanName = betweenName + "_out";
    // Add a splitting node.
    rewriter.setInsertionPointToEnd(nodesOp.getBody());
    std::array<Attribute, 1> inChans{chanOp.varNameAttr()};
    std::array<Attribute, 1> outChans{
        StringAttr{}.get(context, newChanName)};
    rewriter.create<Node>(
        nodesOp.getLoc(), StringAttr{}.get(context, betweenTypeName),
        StringAttr{}.get(context, betweenName), ArrayAttr::get(context, inChans),
        ArrayAttr::get(context, outChans));
    // Split the channel with the node.
    rewriter.setInsertionPointToEnd(chansOp.getBody());
    rewriter.create<Chan>(chansOp.getLoc(), chanOp.typeName(), newChanName,
        BindingAttr{}.get(context, betweenName, outAttr), nodeTo);
    rewriter.replaceOpWithNewOp<Chan>(chanOp, chanOp.typeName(),
        chanOp.varName(), nodeFrom,
        BindingAttr{}.get(context, betweenName, inAttr));
    // Rename target node's input channel.
    nodesOp.walk([&](Node nodeOp) {
      if (nodeOp.name() == nodeTo.getNodeName()) {
        auto &&args = nodeOp.commandArguments();
        std::vector<Attribute> inChans{args.begin(), args.end()};
        for (auto &inChanName : inChans) {
          if (inChanName.cast<StringAttr>().getValue() == chanName) {
            inChanName = StringAttr{}.get(context, newChanName);
            break;
          }
        }
        rewriter.setInsertionPoint(nodeOp);
        rewriter.replaceOpWithNewOp<Node>(
            nodeOp, nodeOp.nodeTypeNameAttr(), nodeOp.nameAttr(),
            ArrayAttr::get(context, inChans), nodeOp.commandResultsAttr());
      }
    });
    return success();
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
      : RewritePattern(mlir::hil::Insts::getOperationName(), 1, context),
        instanceName(instanceName), 
        instanceGraphName(instanceGraphName),
        mainGraphName(mainGraphName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    /// This method is used to notify the rewriter that an in-place operation
    /// modification is about to happen. A call to this function *must* be
    /// followed by a call to either `finalizeRootUpdate` or `cancelRootUpdate`.
    /// This is a minor efficiency win (it avoids creating a new operation and
    /// removing the old one) but also often allows simpler code in the client.
    // virtual void startRootUpdate(Operation *op) {}

    /// This method is used to signal the end of a root update on the given
    /// operation. This can only be called on operations that were provided to a
    /// call to `startRootUpdate`.
    // virtual void finalizeRootUpdate(Operation *op);

    /// This method cancels a pending root update. This can only be called on
    /// operations that were provided to a call to `startRootUpdate`.
    // virtual void cancelRootUpdate(Operation *op) {}

    // MATCH PART.
    //--------------------------------------------------------------------------
    // Get the requested instance.
    auto instOp = dyn_cast<Inst>(op);
    if (!instOp || instOp.name() != instanceName) {
      return failure();
    }
    //--------------------------------------------------------------------------
    // REWRITE PART.
    //--------------------------------------------------------------------------
  
    // Get the main Graph and the instance Graph.
    auto modelOp = cast<mlir::hil::Model>(
        *instOp->getParentOp()->getParentOp()->getParentOp()->getParentOp());
    auto &modelOperations = modelOp.getBody()->getOperations();
    auto graphsOp = findElemByType<Graphs>(modelOperations).value();
    auto graphInstOp = findGraph(graphsOp, instanceGraphName);
    auto graphMainOp = findGraph(graphsOp, mainGraphName);

    // Get the main and the instance graph operations.
    auto &graphInstOperations = graphInstOp->getBody()->getOperations();
    auto &graphMainOperations = graphMainOp->getBody()->getOperations();

    // Get Nodes from the main Graph and the instance Graph.
    auto nodesInstOp = findElemByType<Nodes>(graphInstOperations).value();
    auto nodesMainOp = findElemByType<Nodes>(graphMainOperations).value();

    auto *context = nodesMainOp.getContext();
    // Clone Nodes from the instance graph to the main Graph.
    // NOTE: There may be multiple instances so name change is needed.
    rewriter.setInsertionPointToEnd(nodesMainOp.getBody());
    nodesInstOp.walk([&](Node nodeOp) {
      // Changing the name of the clone Node and 
      // the names of the inputs...
      auto &&args = nodeOp.commandArguments();
      std::vector<Attribute> newInChanNames;
      for (const auto arg : args) {
        auto newInChanName = arg.cast<StringAttr>().str() + "_" 
                                                          + instanceName;
        newInChanNames.push_back(StringAttr{}.get(context, newInChanName));
      }
      // ...and the names of the outputs. 
      auto &&ress = nodeOp.commandResults();
      std::vector<Attribute> newOutChanNames;
      for (const auto res : ress) {
        auto newOutChanName = res.cast<StringAttr>().str() + "_" 
                                                          + instanceName;
        newOutChanNames.push_back(StringAttr{}.get(context, newOutChanName));
      }
      // Creating a new Node with the modified name.
      auto newNodeName = nodeOp.nameAttr().str() + "_" + instanceName;
      rewriter.create<Node>(nodesMainOp.getLoc(),
                            nodeOp.nodeTypeNameAttr(),
                            newNodeName,
                            ArrayAttr::get(context, newInChanNames),
                            ArrayAttr::get(context, newOutChanNames));
    });

    // Get Chans from the main Graph and the instance Graph.
    auto chansInstOp = findElemByType<Chans>(graphInstOperations).value();
    auto chansMainOp = findElemByType<Chans>(graphMainOperations).value();

    // Clone Chans from the instance graph to the main Graph.
    // NOTE: There may be multiple instances so the name changes are needed.
    rewriter.setInsertionPointToEnd(chansMainOp.getBody());
    chansInstOp.walk([&](Chan chanOp) {
      // Creating a new Chan with the modified names.
      auto newChanName = chanOp.varName().str() + "_" + instanceName;
      auto newNodeFromName = chanOp.nodeFromAttr().getNodeName().str() + "_"
                           + instanceName;
      auto newNodeToName = chanOp.nodeToAttr().getNodeName().str() + "_"
                           + instanceName;
      rewriter.create<Chan>(chansMainOp.getLoc(),
                            chanOp.typeName(),
                            newChanName,
                            BindingAttr{}.get(context, newNodeFromName,
                                              chanOp.nodeFrom().getPort()),
                            BindingAttr{}.get(context, newNodeToName,
                                              chanOp.nodeTo().getPort())
                            );
    });

    // Get main Graph connections.
    auto instsMainOp = findElemByType<Insts>(graphMainOperations).value();
    auto reqInstOp = findInst(instsMainOp, instanceName);
    auto &reqInstMainOperations = reqInstOp->getBody()->getOperations();
    auto consReqInsOp = findElemByType<Cons>(reqInstMainOperations).value();

    // Modify the Chans in the main Graph and in the instance Graph.
    std::map<std::string, std::vector<Attribute>> nodeToNewInChanNames;
    std::map<std::string, std::vector<Attribute>> nodeToNewOutChanNames;
    consReqInsOp.walk([&](Con conOp) {
      auto sourceBndGraph = conOp.nodeFromAttr();
      auto targetBndGraph = conOp.nodeToAttr();
      auto dirTypeName = conOp.dirTypeName();
      // Connection to input.
      if (dirTypeName == "IN") {
        auto sourceChanName = sourceBndGraph.getChanName().str();
        auto targetChanNameAfter = targetBndGraph.getChanName().str()
                                    + "_" + instanceName;
        
        auto sourceChanOp = findChan(chansMainOp,
                                      sourceChanName);
        auto targetChanAfterOp = findChan(chansMainOp,
                                           targetChanNameAfter);

        rewriter.setInsertionPointToEnd(chansMainOp.getBody());
        rewriter.replaceOpWithNewOp<Chan>(
           *sourceChanOp,
            sourceChanOp->typeName(),
            sourceChanOp->varName(),
            sourceChanOp->nodeFrom(),
            BindingAttr{}.get(context, 
                targetChanAfterOp->nodeToAttr().getNodeName(),
                targetChanAfterOp->nodeTo().getPort()));
        // Save the name of the input chan to modify it in the node later.
        auto nodeToName = targetChanAfterOp->nodeToAttr().getNodeName().str();
        auto iterator = nodeToNewInChanNames.find(nodeToName);
        std::vector<Attribute> newVectorAttr;
        if (iterator == nodeToNewInChanNames.end()) {
          nodeToNewInChanNames.emplace(nodeToName,
                                       newVectorAttr);
        }
        iterator = nodeToNewInChanNames.find(nodeToName);
        auto newChanNameStr = sourceChanOp->varName().str();
        auto newChanNameAttr = StringAttr{}.get(context, newChanNameStr);
        iterator->second.push_back(newChanNameAttr);
        // Delete the 'source' Node.
        auto sourceNodeNameAfter = 
            targetChanAfterOp->nodeFromAttr().getNodeName().str();
        auto sourceNode = findNode(nodesMainOp, sourceNodeNameAfter);
        rewriter.eraseOp(*sourceNode);
        // Delete the Chan connecting the 'source' Node.
        rewriter.eraseOp(*targetChanAfterOp);
      }

      // Connection to output.
      if (dirTypeName == "OUT") {
        auto targetChanName = targetBndGraph.getChanName().str();
        auto sourceChanNameAfter = sourceBndGraph.getChanName().str() 
                                 + "_" + instanceName;
      
        auto sourceChanAfterOp = findChan(chansMainOp,
                                           sourceChanNameAfter);
        auto targetChanOp = findChan(chansMainOp,
                                      targetChanName);
        rewriter.setInsertionPointToEnd(chansMainOp.getBody());
        rewriter.replaceOpWithNewOp<Chan>(
           *targetChanOp,
            targetChanOp->typeName(),
            targetChanOp->varName(),
            BindingAttr{}.get(context,
                sourceChanAfterOp->nodeFromAttr().getNodeName(),
                sourceChanAfterOp->nodeFrom().getPort()),
            targetChanOp->nodeTo());
        // Save the name of the output chan to modify it in the node later.
        auto nodeFromName = sourceChanAfterOp->nodeFromAttr().getNodeName().str();
        auto iterator = nodeToNewOutChanNames.find(nodeFromName);
        std::vector<Attribute> newVectorAttr;
        if (iterator == nodeToNewOutChanNames.end()) {
          nodeToNewOutChanNames.emplace(nodeFromName,
                                        newVectorAttr);
        }
        iterator = nodeToNewOutChanNames.find(nodeFromName);
        auto newChanNameStr = targetChanOp->varName().str();
        auto newChanNameAttr = StringAttr{}.get(context, newChanNameStr);
        iterator->second.push_back(newChanNameAttr);
        // Delete the 'sink' Node.
        auto sinkNodeNameAfter = 
            sourceChanAfterOp->nodeToAttr().getNodeName().str();
        auto sinkNode = findNode(nodesMainOp, sinkNodeNameAfter);
        rewriter.eraseOp(*sinkNode);
        // Delete the Chan connecting the 'sink' Node.
        rewriter.eraseOp(*sourceChanAfterOp);
      }
      // Delete connection.
      rewriter.eraseOp(conOp);
    });
    rewriter.setInsertionPointToEnd(nodesMainOp.getBody());
    for (const auto &pair : nodeToNewInChanNames) {
      auto nodeOp = findNode(nodesMainOp, pair.first);
      auto inChans = nodeToNewInChanNames.find(pair.first)->second;
      auto outChans = nodeToNewOutChanNames.find(pair.first)->second;
      rewriter.replaceOpWithNewOp<Node>(*nodeOp,
                                         nodeOp->nodeTypeNameAttr(),
                                         nodeOp->name(),
                                         ArrayAttr::get(context, inChans),
                                         ArrayAttr::get(context, outChans));
    }
    // Delete connections containers.
    rewriter.eraseOp(consReqInsOp);
    rewriter.eraseOp(*reqInstOp);

    // Delete the instance node in the main Graph.
    nodesMainOp.walk([&](Node nodeOp) {
      if (nodeOp.name() == instanceName) {
        rewriter.eraseOp(nodeOp);
      }
    });
    return success();
  }

private:
  const std::string instanceName;
  const std::string instanceGraphName;
  const std::string mainGraphName;
};

} // namespace

namespace {
template <typename DerivedT>
class CanonicalizerBase : public ::mlir::OperationPass<> {
public:
  using Base = CanonicalizerBase;

  CanonicalizerBase()
      : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {}
  CanonicalizerBase(const CanonicalizerBase &other)
      : ::mlir::OperationPass<>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral("canonicalize");
  }
  ::llvm::StringRef getArgument() const override { return "canonicalize"; }

  ::llvm::StringRef getDescription() const override {
    return "Canonicalize operations";
  }

  /// Returns the derived pass name.
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral("Canonicalizer");
  }
  ::llvm::StringRef getName() const override { return "Canonicalizer"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

protected:
  ::mlir::Pass::Option<bool> topDownProcessingEnabled{
      *this, "top-down",
      ::llvm::cl::desc("Seed the worklist in general top-down order"),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<bool> enableRegionSimplification{
      *this, "region-simplify",
      ::llvm::cl::desc("Seed the worklist in general top-down order"),
      ::llvm::cl::init(true)};
  ::mlir::Pass::Option<int64_t> maxIterations{
      *this, "max-iterations",
      ::llvm::cl::desc("The max number of PatternSet iterations"),
      ::llvm::cl::init(1)};
  ::mlir::Pass::ListOption<std::string> disabledPatterns{
      *this, "disable-patterns",
      ::llvm::cl::desc(
          "Labels of patterns that should be filtered out during application"),
      llvm::cl::MiscFlags::CommaSeparated};
  ::mlir::Pass::ListOption<std::string> enabledPatterns{
      *this, "enable-patterns",
      ::llvm::cl::desc("Labels of patterns that should be used during "
                       "application, all other patterns are filtered out"),
      llvm::cl::MiscFlags::CommaSeparated};
};

struct GraphCanonicalizer : public CanonicalizerBase<GraphCanonicalizer> {
  GraphCanonicalizer(const GreedyRewriteConfig &config) : config(config) {}

  GraphCanonicalizer() {
    // Default constructed GraphCanonicalizer takes its settings from command
    // line options.
    config.useTopDownTraversal = topDownProcessingEnabled;
    config.enableRegionSimplification = enableRegionSimplification;
    config.maxIterations = maxIterations;
  }

  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    owningPatterns.add<ChansRewritePass>(context);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    (void)applyPatternsAndFoldGreedily(getOperation()->getRegions(), patterns,
                                       config);
  }

  GreedyRewriteConfig config;
  FrozenRewritePatternSet patterns;
};

} // namespace

namespace mlir::transforms {
void runPass(MLIRModule &mlirModule, RewritePattern &&pass) {
  auto *context = mlirModule.getContext();
  SimpleRewriter rewriter(context);
  mlirModule.getRoot()->walk(
      [&](Operation *op) { (void)pass.matchAndRewrite(op, rewriter); });
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