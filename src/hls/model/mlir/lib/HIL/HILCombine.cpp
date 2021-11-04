#include "HIL/HILCombine.h"
#include "HIL/HILDialect.h"
#include "HIL/HILOps.h"
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
class ChanRewritePattern : public RewritePattern {
  public:
    ChanRewritePattern(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

    LogicalResult match(Operation *op) const override { return success(); }

    void rewrite(Operation *chan_op, PatternRewriter &rewriter) const override {
        if (isa<Chan>(chan_op) && chan_op->getAttrs().size() != 4) {
            auto context = chan_op->getContext();
            auto region = chan_op->getParentRegion();
            auto chan_name =
                chan_op->getAttrOfType<StringAttr>("varName").getValue().str();
            Optional<StringAttr> nodeFrom = {};
            Optional<StringAttr> nodeTo = {};
            for (auto &node_op : region->getOps()) {
                if (isa<Node>(node_op)) {
                    for (auto in_chan_op :
                         node_op.getAttrOfType<ArrayAttr>("commandArguments")) {
                        auto in_chan_name =
                            in_chan_op.cast<StringAttr>().getValue().str();
                        if (in_chan_name == chan_name) {
                            nodeTo = node_op.getAttrOfType<StringAttr>("name");
                        }
                    }
                    for (auto out_chan_op :
                         node_op.getAttrOfType<ArrayAttr>("commandResults")) {
                        auto out_chan_name =
                            out_chan_op.cast<StringAttr>().getValue().str();
                        if (out_chan_name == chan_name) {
                            nodeFrom = node_op.getAttrOfType<StringAttr>("name");
                        }
                    }
                }
            }
            rewriter.replaceOpWithNewOp<Chan>(
                chan_op, chan_op->getAttrOfType<StringAttr>("typeName"),
                chan_op->getAttrOfType<StringAttr>("varName"),
                nodeFrom.getValueOr(StringAttr().get(context, "UNK")),
                nodeTo.getValueOr(StringAttr().get(context, "UNK")));
        }
    }
};
} // namespace

namespace {
template <typename DerivedT>
class CanonicalizerBase : public ::mlir::OperationPass<> {
  public:
    using Base = CanonicalizerBase;

    CanonicalizerBase() : ::mlir::OperationPass<>(::mlir::TypeID::get<DerivedT>()) {}
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
        owningPatterns.add<ChanRewritePattern>(context);

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
