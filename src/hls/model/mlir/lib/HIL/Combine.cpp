#include "HIL/Combine.h"
#include "HIL/Dialect.h"
#include "HIL/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

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
    auto chans_op = dyn_cast<Chans>(*op);
    if (!chans_op) {
      return failure();
    }
    auto context = chans_op.getContext();
    auto outer_region_ops = chans_op->getParentRegion()->getOps();
    auto nodes_op_it =
        std::find_if(outer_region_ops.begin(), outer_region_ops.end(),
                     [](Operation &op) { return isa<Nodes>(op); });
    assert(nodes_op_it != outer_region_ops.end());
    auto nodes_op = cast<Nodes>(*nodes_op_it);
    std::map<std::string, std::string> chan_to_source;
    std::map<std::string, std::string> chan_to_target;
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
    std::vector<std::reference_wrapper<Operation>> ops;
    std::copy(chans_op.getBody()->getOperations().begin(),
              chans_op.getBody()->getOperations().end(),
              std::back_inserter(ops));
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
          StringAttr().get(context, node_from.value()),
          StringAttr().get(context, node_to.value()));
    }
    return success();
  }
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
      ::llvm::cl::init(10)};
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

std::unique_ptr<Pass> createGraphRewritePass() {
  return std::make_unique<GraphCanonicalizer>();
}
