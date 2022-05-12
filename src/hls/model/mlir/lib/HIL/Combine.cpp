//===- Combine.cpp - MLIR passes -----------------*- C++ -*----------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

using namespace mlir;
using namespace mlir::hil;

namespace {
class ChansRewritePass : public RewritePattern {
public:
  ChansRewritePass(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    // detect channels container operation
    auto chans_op = dyn_cast<Chans>(*op);
    if (!chans_op) {
      return failure();
    }
    auto context = chans_op.getContext();
    auto outer_region_ops = chans_op->getParentRegion()->getOps();

    // find Nodes operation
    auto nodes_op = find_elem_by_type<Nodes>(outer_region_ops).value();

    // in_chan_name->node_name map
    std::map<std::string, std::string> chan_to_source;

    // out_chan_name->node_name map
    std::map<std::string, std::string> chan_to_target;

    // iterate over Nodes' sub-operations, fill maps
    for (auto &nodes_block_op : nodes_op.getBody()->getOperations()) {

      auto node_op = cast<Node>(nodes_block_op);
      auto node_name = node_op.name().str();

      for (auto in_chan_op : node_op.commandArguments()) {

        auto in_chan_name = in_chan_op.cast<StringAttr>().getValue().str();
        chan_to_target[in_chan_name] = node_name;
      }

      for (auto out_chan_op : node_op.commandResults()) {

        auto out_chan_name = out_chan_op.cast<StringAttr>().getValue().str();
        chan_to_source[out_chan_name] = node_name;
      }
    }

    // copy channels into vector
    std::vector<std::reference_wrapper<Operation>> ops;
    std::copy(chans_op.getBody()->getOperations().begin(),
              chans_op.getBody()->getOperations().end(),
              std::back_inserter(ops));

    // iterate over channels
    for (auto &chans_block_op_ref : ops) {

      auto &chans_block_op = chans_block_op_ref.get();
      auto chan_op = cast<Chan>(chans_block_op);
      auto chan_name = chan_op.varName().str();

      std::optional<std::string> node_from;
      auto node_from_it = chan_to_source.find(chan_name);

      if (node_from_it != chan_to_source.end()) {
        node_from = node_from_it->second;
      }

      std::optional<std::string> node_to;
      auto node_to_it = chan_to_target.find(chan_name);

      if (node_to_it != chan_to_target.end()) {
        node_to = node_to_it->second;
      }

      rewriter.setInsertionPoint(&chans_block_op);
      rewriter.replaceOpWithNewOp<Chan>(
          &chans_block_op, chan_op.typeName(), chan_op.varName(),
          StringAttr{}.get(context, node_from.value()),
          StringAttr{}.get(context, node_to.value()));
    }
    return success();
  }
};

class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

class InsertDelayPass : public RewritePattern {
public:
  InsertDelayPass(MLIRContext *context, std::string chan_name, unsigned latency)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        chan_name_(chan_name), latency_(latency) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    // detect requested channel
    auto chan_op = dyn_cast<Chan>(*op);
    if (!chan_op || chan_op.varName() != chan_name_) {
      return failure();
    }
    auto chans_op = cast<Chans>(*chan_op->getParentOp());
    auto model_op = cast<Model>(*chans_op->getParentOp()->getParentOp());
    auto &model_ops = model_op.getBody()->getOperations();
    auto nodetypes_op = find_elem_by_type<NodeTypes>(model_ops).value();
    auto graph_op = find_elem_by_type<Graph>(model_ops).value();
    auto &graph_ops = graph_op.getBody()->getOperations();
    auto nodes_op = find_elem_by_type<Nodes>(graph_ops).value();
    auto n_nodes = nodes_op.getBody()->getOperations().size();

    auto chan_type = chan_op.typeName();
    auto node_from = chan_op.nodeFromAttr();
    auto node_to = chan_op.nodeToAttr();

    auto btw_type_name =
        "delay_" + chan_type.str() + "_" + std::to_string(latency_);
    auto btw_name = btw_type_name + "_" + std::to_string(n_nodes);

    // Check if we already added a delay
    bool already_added_delay = false;
    nodetypes_op.walk([&](NodeType op) {
      if (!already_added_delay && op.name() == btw_type_name) {
        already_added_delay = true;
      }
    });
    if (already_added_delay) {
      return failure();
    }

    auto context = nodetypes_op.getContext();
    // Add nodetype
    auto in_attr =
        InputPortAttr::get(context, chan_type.str(), new double{1.0}, "in");
    auto out_attr = OutputPortAttr::get(context, chan_type.str(),
                                       new double{1.0}, latency_, "out", "0");
    std::array<Attribute, 1> in_attrs{in_attr};
    std::array<Attribute, 1> out_attrs{out_attr};
    rewriter.setInsertionPointToEnd(nodetypes_op.getBody());
    rewriter.create<NodeType>(
        nodetypes_op.getLoc(), StringAttr{}.get(context, btw_type_name),
        ArrayAttr::get(context, in_attrs), ArrayAttr::get(context, out_attrs));
    auto new_chan_name = btw_name + "_out";
    // Add a splitting node
    rewriter.setInsertionPointToEnd(nodes_op.getBody());
    std::array<Attribute, 1> in_chans{chan_op.varNameAttr()};
    std::array<Attribute, 1> out_chans{
        StringAttr{}.get(context, new_chan_name)};
    rewriter.create<Node>(
        nodetypes_op.getLoc(), StringAttr{}.get(context, btw_type_name),
        StringAttr{}.get(context, btw_name), ArrayAttr::get(context, in_chans),
        ArrayAttr::get(context, out_chans));
    // Split the channel with the node
    rewriter.setInsertionPointToEnd(chans_op.getBody());
    rewriter.create<Chan>(chans_op.getLoc(), chan_op.typeName(), new_chan_name,
                          StringAttr{}.get(context, btw_name), node_to.getNodeName());
    rewriter.replaceOpWithNewOp<Chan>(chan_op, chan_op.typeName(),
                                      chan_op.varName(), node_from.getNodeName(),
                                      StringAttr{}.get(context, btw_name));
    // Rename target node's input channel
    nodes_op.walk([&](Node op) {
      if (op.name() == node_to.getNodeName()) {
        auto &&args = op.commandArguments();
        std::vector<Attribute> in_chans{args.begin(), args.end()};
        for (auto &in_chan_name : in_chans) {
          if (in_chan_name.cast<StringAttr>().getValue() == chan_name_) {
            in_chan_name = StringAttr{}.get(context, new_chan_name);
            break;
          }
        }
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<Node>(
            op, op.nodeTypeNameAttr(), op.nameAttr(),
            ArrayAttr::get(context, in_chans), op.commandResultsAttr());
      }
    });
    return success();
  }

private:
  const std::string chan_name_;
  const unsigned latency_;
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
      ::llvm::cl::desc("Seed the worklist in general top-down order"),
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
void run_pass(MLIRModule &m, RewritePattern &&pass) {
  auto context = m.get_context();
  SimpleRewriter rewriter(context);
  m.get_root()->walk(
      [&](Operation *op) { (void)pass.matchAndRewrite(op, rewriter); });
}

std::function<void(MLIRModule &)> ChanAddSourceTarget() {
  return [=](MLIRModule &m) {
    auto context = m.get_context();
    ChansRewritePass pass{context};
    run_pass(m, std::move(pass));
  };
}

std::function<void(MLIRModule &)> InsertDelay(std::string chan_name,
                                              unsigned latency) {
  return [=](MLIRModule &m) {
    auto context = m.get_context();
    InsertDelayPass pass{context, chan_name, latency};
    run_pass(m, std::move(pass));
  };
}
} // namespace mlir::transforms

std::unique_ptr<Pass> createGraphRewritePass() {
  return std::make_unique<GraphCanonicalizer>();
}
