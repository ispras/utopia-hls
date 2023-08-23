//===- Conversion.cpp - Translate HIL into FIRRTL --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main HIL to FIRRTL Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "HIL/API.h"
#include "HIL/Conversion.h"
#include "HIL/Dialect.h"
#include "HIL/Model.h"
#include "HIL/Ops.h"
#include "HIL/Utils.h"

#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"

#include "mlir/Transforms/DialectConversion.h"

template<typename Type>
using ArrayRef = mlir::ArrayRef<Type>;
using ChansOp = mlir::hil::ChansOp;
using ChanOp = mlir::hil::ChanOp;
using ConsOp = mlir::hil::ConsOp;
using ConOp = mlir::hil::ConOp;
using ConversionPattern = mlir::ConversionPattern;
using ConversionPatternRewriter = mlir::ConversionPatternRewriter;
using ConversionTarget = mlir::ConversionTarget;
using ConvertedOps = mlir::DenseSet<mlir::Operation *>;
using DialectRegistry = mlir::DialectRegistry;
using FIRRTLDialect = circt::firrtl::FIRRTLDialect;
using HILDialect = mlir::hil::HILDialect;
using LLVMStringLiteral = llvm::StringLiteral;
using LLVMStringRef = llvm::StringRef;
using LogicalResult = mlir::LogicalResult;
using MLIRContext = mlir::MLIRContext;
using ModelOp = mlir::hil::ModelOp;
using ModuleOp = mlir::ModuleOp;
using NodeTypesOp = mlir::hil::NodeTypesOp;
using NodeTypeOp = mlir::hil::NodeTypeOp;
using NodesOp = mlir::hil::NodesOp;
using NodeOp = mlir::hil::NodeOp;
template<typename OperationType>
using OpConversionPattern = mlir::OpConversionPattern<OperationType>;
using Operation = mlir::Operation;
template<typename Type = void>
using OperationPass = mlir::OperationPass<Type>;
using Pass = mlir::Pass;
using RewritePattern = mlir::RewritePattern;
using RewritePatternSet = mlir::RewritePatternSet;
using TypeConverter = mlir::TypeConverter;
using TypeID = mlir::TypeID;
using Value = mlir::Value;

namespace {

class FIRRTLTypeConverter : public TypeConverter {
public:
  FIRRTLTypeConverter() {
    /// TODO:
    // addConversion([](Type type) -> Type {
    //   if (type.isa<NoneType>())
    //     return dc::TokenType::get(type.getContext());
    //   return dc::ValueType::get(type.getContext(), type);
    // });
    // addConversion([](ValueType type) { return type; });
    // addConversion([](TokenType type) { return type; });

    // addTargetMaterialization(
    //     [](mlir::OpBuilder &builder, mlir::Type resultType,
    //        mlir::ValueRange inputs,
    //        mlir::Location loc) -> std::optional<mlir::Value> {
    //       /// TODO:
    //       // if (inputs.size() != 1)
    //       //   return std::nullopt;

    //       // // Materialize !dc.value<> -> !dc.token
    //       // if (resultType.isa<dc::TokenType>() &&
    //       //     inputs.front().getType().isa<dc::ValueType>())
    //       //   return unpack(builder, inputs.front()).token;

    //       // // Materialize !dc.token -> !dc.value<>
    //       // auto vt = resultType.dyn_cast<dc::ValueType>();
    //       // if (vt && vt.getInnerTypes().empty())
    //       //   return pack(builder, inputs.front(), ValueRange{});

    //       return inputs[0];
    //     });

    // addSourceMaterialization(
    //     [](mlir::OpBuilder &builder, mlir::Type resultType,
    //        mlir::ValueRange inputs,
    //        mlir::Location loc) -> std::optional<mlir::Value> {
    //       /// TODO:
    //       // if (inputs.size() != 1)
    //       //   return std::nullopt;

    //       // // Materialize !dc.value<> -> !dc.token
    //       // if (resultType.isa<dc::TokenType>() &&
    //       //     inputs.front().getType().isa<dc::ValueType>())
    //       //   return unpack(builder, inputs.front()).token;

    //       // // Materialize !dc.token -> !dc.value<>
    //       // auto vt = resultType.dyn_cast<dc::ValueType>();
    //       // if (vt && vt.getInnerTypes().empty())
    //       //   return pack(builder, inputs.front(), ValueRange{});

    //       return inputs[0];
    //     });
  }
};

template <typename OperationType>
class FIRRTLOpConversionPattern : public OpConversionPattern<OperationType> {
public:
  using OpConversionPattern<OperationType>::OpConversionPattern;
  using OpAdaptor = typename OperationType::Adaptor;

  FIRRTLOpConversionPattern(MLIRContext *context, TypeConverter &typeConverter,
                            ConvertedOps *convertedOps)
      : OpConversionPattern<OperationType>(typeConverter, context),
        convertedOps(convertedOps) {}
  mutable ConvertedOps *convertedOps;
};

class ModelOpConversionPattern : public FIRRTLOpConversionPattern<ModelOp> {
public:
  using FIRRTLOpConversionPattern<ModelOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ModelOp::Adaptor;

  LogicalResult matchAndRewrite(ModelOp modelOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class GraphsOpConversionPattern : public FIRRTLOpConversionPattern<GraphsOp> {
public:
  using FIRRTLOpConversionPattern<GraphsOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename GraphsOp::Adaptor;

  LogicalResult matchAndRewrite(GraphsOp graphsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class GraphOpConversionPattern : public FIRRTLOpConversionPattern<GraphOp> {
public:
  using FIRRTLOpConversionPattern<GraphOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename GraphOp::Adaptor;

  LogicalResult matchAndRewrite(GraphOp graphOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class NodeTypesOpConversionPattern :
    public FIRRTLOpConversionPattern<NodeTypesOp> {
public:
  using FIRRTLOpConversionPattern<NodeTypesOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename NodeTypesOp::Adaptor;

  LogicalResult matchAndRewrite(NodeTypesOp nodeTypesOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class NodeTypeOpConversionPattern :
    public FIRRTLOpConversionPattern<NodeTypeOp> {
public:
  using FIRRTLOpConversionPattern<NodeTypeOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename NodeTypeOp::Adaptor;

  LogicalResult matchAndRewrite(NodeTypeOp nodeTypeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class NodesOpConversionPattern : public FIRRTLOpConversionPattern<NodesOp> {
public:
  using FIRRTLOpConversionPattern<NodesOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename NodesOp::Adaptor;

  LogicalResult matchAndRewrite(NodesOp nodesOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class NodeOpConversionPattern : public FIRRTLOpConversionPattern<NodeOp> {
public:
  using FIRRTLOpConversionPattern<NodeOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename NodeOp::Adaptor;

  LogicalResult matchAndRewrite(NodeOp nodeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class ChansOpConversionPattern : public FIRRTLOpConversionPattern<ChansOp> {
public:
  using FIRRTLOpConversionPattern<ChansOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ChansOp::Adaptor;

  LogicalResult matchAndRewrite(ChansOp chansOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class ChanOpConversionPattern : public FIRRTLOpConversionPattern<ChanOp> {
public:
  using FIRRTLOpConversionPattern<ChanOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ChanOp::Adaptor;

  LogicalResult matchAndRewrite(ChanOp chanOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class ConsOpConversionPattern : public FIRRTLOpConversionPattern<ConsOp> {
public:
  using FIRRTLOpConversionPattern<ConsOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ConsOp::Adaptor;

  LogicalResult matchAndRewrite(ConsOp consOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

class ConOpConversionPattern : public FIRRTLOpConversionPattern<ConOp> {
public:
  using FIRRTLOpConversionPattern<ConOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ConOp::Adaptor;

  LogicalResult matchAndRewrite(ConOp conOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO:
    return mlir::success();
  }
};

template <typename DerivedT>
class HILToFIRRTLBase : public OperationPass<ModuleOp> {
public:
  using Base = HILToFIRRTLBase;

  HILToFIRRTLBase()
      : OperationPass<ModuleOp>(TypeID::get<DerivedT>()) {}
  HILToFIRRTLBase(const HILToFIRRTLBase &other)
      : OperationPass<ModuleOp>(other) {}

  /// Returns the command-line argument attached to this pass.
  static constexpr LLVMStringLiteral getArgumentName() {
    return LLVMStringLiteral("lower-hil-to-firrtl");
  }
  LLVMStringRef getArgument() const override { return "lower-hil-to-firrtl"; }

  LLVMStringRef getDescription() const override {
    return "Lower HIL to FIRRTL";
  }

  /// Returns the derived pass name.
  static constexpr LLVMStringLiteral getPassName() {
    return LLVMStringLiteral("HILToFIRRTL");
  }
  LLVMStringRef getName() const override { return "HILToFIRRTL"; }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const Pass *pass) {
    return pass->getTypeID() == TypeID::get<DerivedT>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<DerivedT>(*static_cast<const DerivedT *>(this));
  }

  /// Return the dialect that must be loaded in the context before this pass.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FIRRTLDialect>();
  }
};

class HILToFIRRTLPass : public HILToFIRRTLBase<HILToFIRRTLPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    ConvertedOps convertedOps;

    ConversionTarget target(getContext());
    target.addIllegalDialect<HILDialect>();
    target.addLegalDialect<FIRRTLDialect>();
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return convertedOps.contains(op); });

    FIRRTLTypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    patterns
        .add<ModelOpConversionPattern, GraphsOpConversionPattern,
             GraphOpConversionPattern, NodeTypesOpConversionPattern,
             NodeTypeOpConversionPattern, NodesOpConversionPattern,
             NodeOpConversionPattern, ChansOpConversionPattern,
             ChanOpConversionPattern, ConsOpConversionPattern,
             ConOpConversionPattern>(
            &getContext(), typeConverter, &convertedOps);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> createHILToFIRRTLPass() {
  return std::make_unique<HILToFIRRTLPass>();
}