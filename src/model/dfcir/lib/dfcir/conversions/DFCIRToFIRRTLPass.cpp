//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "dfcir/conversions/ModuleDefines.inc"

namespace circt::firrtl::utils {

Value getBlockArgument(Block *block, unsigned ind) {
  return block->getArgument(ind);
}

Value getBlockArgumentFromOpBlock(Operation *op, unsigned ind) {
  return getBlockArgument(op->getBlock(), ind);
}

Value getClockVar(Block *block) {
  Value arg = getBlockArgument(block, block->getNumArguments() - 1);
  if (arg.getType().isa<ClockType>()) {
    return arg;
  }
  return nullptr;
}

Value getClockVarFromOpBlock(Operation *op) {
  return getClockVar(op->getBlock());
}

} // namespace circt::firrtl::utils

template <>
struct std::hash<std::pair<mlir::Operation *, unsigned>> {
  size_t operator()(
      const std::pair<mlir::Operation *, unsigned> &pair) const noexcept {
    return std::hash<mlir::Operation *>()(pair.first) +
           std::hash<unsigned>()(pair.second);
  }
};

namespace mlir::dfcir {

#define GEN_PASS_DECL_DFCIRTOFIRRTLPASS
#define GEN_PASS_DEF_DFCIRTOFIRRTLPASS

#include "dfcir/conversions/DFCIRPasses.h.inc"

class FIRRTLTypeConverter : public TypeConverter {
public:
  FIRRTLTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](DFCIRConstantType type) -> circt::firrtl::IntType {
      Type constType = type.getConstType();
      if (constType.isa<DFCIRFixedType>()) {
        DFCIRFixedType fixedType = llvm::cast<DFCIRFixedType>(constType);
        unsigned width =
            fixedType.getIntegerBits() + fixedType.getFractionBits();
        if (fixedType.getSign()) {
          return circt::firrtl::SIntType::get(fixedType.getContext(), width);
        } else {
          return circt::firrtl::UIntType::get(fixedType.getContext(), width);
        }
      } else if (constType.isa<DFCIRFloatType>()) {
        DFCIRFloatType floatType = llvm::cast<DFCIRFloatType>(constType);
        unsigned width =
            floatType.getExponentBits() + floatType.getFractionBits();
        return circt::firrtl::UIntType::get(floatType.getContext(), width);
      }
      return {};
    });
    addConversion([](DFCIRScalarType type) -> circt::firrtl::IntType {
      Type scalarType = type.getScalarType();
      if (scalarType.isa<DFCIRFixedType>()) {
        DFCIRFixedType fixedType = llvm::cast<DFCIRFixedType>(scalarType);
        unsigned width =
            fixedType.getIntegerBits() + fixedType.getFractionBits();
        if (fixedType.getSign()) {
          return circt::firrtl::SIntType::get(fixedType.getContext(), width,
                                              true);
        } else {
          return circt::firrtl::UIntType::get(fixedType.getContext(), width,
                                              true);
        }
      } else if (scalarType.isa<DFCIRFloatType>()) {
        DFCIRFloatType floatType = llvm::cast<DFCIRFloatType>(scalarType);
        unsigned width =
            floatType.getExponentBits() + floatType.getFractionBits();
        return circt::firrtl::UIntType::get(floatType.getContext(), width);
      }
      return {};
    });
    addConversion([](DFCIRStreamType type) -> circt::firrtl::IntType {
      Type streamType = type.getStreamType();
      if (streamType.isa<DFCIRFixedType>()) {
        DFCIRFixedType fixedType = llvm::cast<DFCIRFixedType>(streamType);
        unsigned width =
            fixedType.getIntegerBits() + fixedType.getFractionBits();
        if (fixedType.getSign()) {
          return circt::firrtl::SIntType::get(fixedType.getContext(), width);
        } else {
          return circt::firrtl::UIntType::get(fixedType.getContext(), width);
        }
      } else if (streamType.isa<DFCIRFloatType>()) {
        DFCIRFloatType floatType = llvm::cast<DFCIRFloatType>(streamType);
        unsigned width =
            floatType.getExponentBits() + floatType.getFractionBits();
        return circt::firrtl::UIntType::get(floatType.getContext(), width);
      }
      return {};
    });
  }
};

template <typename OperationType>
class FIRRTLOpConversionPattern : public OpConversionPattern<OperationType> {
public:
  using OpConversionPattern<OperationType>::OpConversionPattern;
  using ConvertedOps = mlir::DenseSet<mlir::Operation *>;
  using OldTypeMap =
      std::unordered_map<std::pair<mlir::Operation *, unsigned>, mlir::Type>;

protected:
  mutable ConvertedOps *convertedOps;
  // TODO: Change when replaceAllUsesWith-related 
  // pull request for MLIR is approved.
  // Issue #17 (https://github.com/ispras/utopia-hls/issues/17).
  OldTypeMap *oldTypeMap;   
  ModuleArgMap *moduleArgMap;

public:
  FIRRTLOpConversionPattern(MLIRContext *context,
                            TypeConverter &typeConverter,
                            ConvertedOps *convertedOps,
                            OldTypeMap *oldTypeMap,
                            ModuleArgMap *moduleArgMap)
    : OpConversionPattern<OperationType>(typeConverter, context),
      convertedOps(convertedOps),
      oldTypeMap(oldTypeMap),
      moduleArgMap(moduleArgMap) {
    // Required to allow root updates, which imply recursive
    // pattern application.
    this->setHasBoundedRewriteRecursion(true);
  }
};

class KernelOpConversionPattern : public FIRRTLOpConversionPattern<KernelOp> {
public:
  using FIRRTLOpConversionPattern<KernelOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename KernelOp::Adaptor;
  using CircuitOp = circt::firrtl::CircuitOp;
  using FModuleOp = circt::firrtl::FModuleOp;
  using ConventionAttr = circt::firrtl::ConventionAttr;
  using InputOp = mlir::dfcir::InputOp;
  using OutputOp = mlir::dfcir::OutputOp;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    Block *kernelBlock = &(kernelOp.getBody().front());
    auto save = rewriter.saveInsertionPoint();

    // Create a new circuit to substitute the kernel with.
    auto circuitOp = rewriter.create<CircuitOp>(
      kernelOp.getLoc(),
      rewriter.getStringAttr(kernelOp.getName()));

    // Collect info on inputs and outputs.
    SmallVector<Operation *> ports;
    SmallVector<circt::firrtl::PortInfo> modulePorts;
    unsigned argInd = 0;
    for (Operation &op: kernelBlock->getOperations()) {
      if (auto named = llvm::dyn_cast<NamedOpVal>(op)) {
        (*moduleArgMap)[&op] = argInd++;
        llvm::StringRef name = named.getValueName();
        ports.push_back(&op);
        Type converted =
            getTypeConverter()->convertType(op.getResult(0).getType());
        modulePorts.emplace_back(
          mlir::StringAttr::get(getContext(), name),
          converted,
          (llvm::isa<InputOpInterface>(op))
            ? circt::firrtl::Direction::In
            : circt::firrtl::Direction::Out);
      }
    }

    // Add explicit clock argument.
    modulePorts.emplace_back(
      mlir::StringAttr::get(rewriter.getContext(), CLOCK_ARG),
      circt::firrtl::ClockType::get(rewriter.getContext()),
      circt::firrtl::Direction::In);

    // Add a module to represent the old kernel with.
    rewriter.setInsertionPointToStart(circuitOp.getBodyBlock());
    auto fModuleOp =
        rewriter.create<FModuleOp>(rewriter.getUnknownLoc(),
                                   StringAttr::get(rewriter.getContext(),
                                   kernelOp.getName()),
                                   ConventionAttr::get(
                                       rewriter.getContext(),
                                       circt::firrtl::Convention::Internal),
                                   modulePorts);

    // Replace the input-/output-operations' results with block arguments.
    for (size_t index = 0; index < ports.size(); ++index) {
      BlockArgument arg = fModuleOp.getArgument(index);
      for (auto &operand: llvm::make_early_inc_range(
          ports[index]->getResult(0).getUses())) {
        (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                     operand.getOperandNumber())] =
                                         operand.get().getType();
        operand.set(arg);
      }
    }

    // Empty arguments assumed.
    rewriter.mergeBlocks(kernelBlock, fModuleOp.getBodyBlock());
    rewriter.restoreInsertionPoint(save);
    rewriter.replaceOp(kernelOp, circuitOp);
    return mlir::success();
  }
};

class InputOpConversionPattern : public FIRRTLOpConversionPattern<InputOp> {
public:
  using FIRRTLOpConversionPattern<InputOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename InputOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(InputOp inputOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    // TODO: Add control stream functionality.
    // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
    rewriter.eraseOp(inputOp);
    return mlir::success();
  }
};

class ScalarInputOpConversionPattern
    : public FIRRTLOpConversionPattern<ScalarInputOp> {
public:
  using FIRRTLOpConversionPattern<ScalarInputOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ScalarInputOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(ScalarInputOp scalarInputOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    rewriter.eraseOp(scalarInputOp);
    return mlir::success();
  }
};

class OutputOpConversionPattern : public FIRRTLOpConversionPattern<OutputOp> {
public:
  using FIRRTLOpConversionPattern<OutputOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename OutputOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(OutputOp outputOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::utils::getBlockArgumentFromOpBlock;

    // TODO: Add control stream functionality.
    // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
    if (outputOp.getStream()) {
      rewriter.create<circt::firrtl::ConnectOp>(
          rewriter.getUnknownLoc(),
          getBlockArgumentFromOpBlock(outputOp, (*moduleArgMap)[outputOp]),
          adaptor.getStream()
      );
    }
    rewriter.eraseOp(outputOp);
    return mlir::success();
  }
};

class ScalarOutputOpConversionPattern
    : public FIRRTLOpConversionPattern<ScalarOutputOp> {
public:
  using FIRRTLOpConversionPattern<ScalarOutputOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ScalarOutputOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(ScalarOutputOp scalarOutputOp,
                                OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::utils::getBlockArgumentFromOpBlock;

    // TODO: Add control stream functionality.
    // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
    if (scalarOutputOp.getStream()) {
      rewriter.create<circt::firrtl::ConnectOp>(
          rewriter.getUnknownLoc(),
          getBlockArgumentFromOpBlock(scalarOutputOp,
                                      (*moduleArgMap)[scalarOutputOp]),
          adaptor.getStream()
      );
    }
    rewriter.eraseOp(scalarOutputOp);
    return mlir::success();
  }
};

class ConstantOpConversionPattern
    : public FIRRTLOpConversionPattern<ConstantOp> {
public:
  using FIRRTLOpConversionPattern<ConstantOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ConstantOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(ConstantOp constOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::ConstantOp;
    using circt::firrtl::UIntType;
    using circt::firrtl::SIntType;
    using circt::firrtl::IntType;
    auto castedInt = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
    auto castedFloat = llvm::dyn_cast<mlir::FloatAttr>(constOp.getValue());
    Type newType = getTypeConverter()->convertType(constOp.getRes().getType());
    circt::firrtl::ConstantOp newOp;

    if (castedInt) {
      newOp = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(),
                                          newType, castedInt);
    } else if (castedFloat) {
      // TODO: Add float functionality.
      // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
      assert(false && "No floats yet");
    }
    for (auto &operand: llvm::make_early_inc_range(
        constOp->getResult(0).getUses())) {
      (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                   operand.getOperandNumber())] =
                                       operand.get().getType();
      operand.set(newOp.getResult());
    }
    rewriter.eraseOp(constOp);
    return mlir::success();
  }
};

SmallVector<circt::firrtl::PortInfo> getUnaryOpPorts(mlir::Type outType,
                                                     mlir::Type firstType,
                                                     mlir::MLIRContext *ctx) {
  return SmallVector<circt::firrtl::PortInfo>  {
    circt::firrtl::PortInfo(
        mlir::StringAttr::get(ctx, "res1"),
        outType,
        circt::firrtl::Direction::Out),
    circt::firrtl::PortInfo(
        mlir::StringAttr::get(ctx, "arg1"),
        firstType,
        circt::firrtl::Direction::In),
    circt::firrtl::PortInfo(
        mlir::StringAttr::get(ctx, "clk"),
        circt::firrtl::ClockType::get(ctx),
        circt::firrtl::Direction::In)
  };
}

SmallVector<circt::firrtl::PortInfo> getBinaryOpPorts(mlir::Type outType,
                                                      mlir::Type firstType,
                                                      mlir::Type secondType,
                                                      mlir::MLIRContext *ctx) {
  return SmallVector<circt::firrtl::PortInfo>  {
    circt::firrtl::PortInfo(
        mlir::StringAttr::get(ctx, "res1"),
        outType,
        circt::firrtl::Direction::Out),
    circt::firrtl::PortInfo(
        mlir::StringAttr::get(ctx, "arg1"),
        firstType,
        circt::firrtl::Direction::In),
    circt::firrtl::PortInfo(
        mlir::StringAttr::get(ctx, "arg2"),
        secondType,
        circt::firrtl::Direction::In),
    circt::firrtl::PortInfo(
        mlir::StringAttr::get(ctx, "clk"),
        circt::firrtl::ClockType::get(ctx),
        circt::firrtl::Direction::In)
  };
}

template <typename OperationType, typename AdaptorType>
class SchedulableOpConversionPattern 
    : public FIRRTLOpConversionPattern<OperationType> {
  using FExtModuleOp = circt::firrtl::FExtModuleOp;
  using InstanceOp = circt::firrtl::InstanceOp;
  using CircuitOp = circt::firrtl::CircuitOp;
  using Rewriter = ConversionPatternRewriter;

  virtual std::string
  getBaseModuleName() const = 0;

  virtual std::string
  constructModuleName(OperationType &op, AdaptorType &adaptor) const {
    std::string name = getBaseModuleName();
    llvm::raw_string_ostream nameStream(name);
  
    for (unsigned id = 0; id < op->getNumOperands(); ++id) {
      nameStream << "_IN_";
      Type oldType = (*this->oldTypeMap)[std::make_pair(op, id)];
      Type innerType = llvm::cast<DFType>(oldType).getDFType();
      llvm::cast<SVSynthesizable>(innerType).printSVSignature(nameStream);
    }

    for (unsigned id = 0; id < op->getNumResults(); ++id) {
      nameStream << "_OUT_";
      auto res = op->getResult(id);
      Type oldType = res.getType();
      Type innerType = llvm::cast<DFType>(oldType).getDFType();
      llvm::cast<SVSynthesizable>(innerType).printSVSignature(nameStream);
    }

    nameStream << "_" << llvm::cast<Scheduled>(op.getOperation()).getLatency();

    return name;
  }

  virtual FExtModuleOp
  createModule(const std::string &name, OperationType &op,
               AdaptorType &adaptor, Rewriter &rewriter) const {
    Type type = op->getResult(0).getType();
    Type convertedType = this->getTypeConverter()->convertType(type);

    SmallVector<circt::firrtl::PortInfo> ports;
    auto adaptorOperands = adaptor.getOperands();
    if (op->getNumOperands() == 1) {
      ports = getUnaryOpPorts(convertedType,
                              adaptorOperands[0].getType(),
                              rewriter.getContext());
    } else {
      ports = getBinaryOpPorts(convertedType,
                               adaptorOperands[0].getType(),
                               adaptorOperands[1].getType(),
                               rewriter.getContext());
    }

    IntegerType attrType = mlir::IntegerType::get(rewriter.getContext(), 32,
                                                  mlir::IntegerType::Unsigned);

    int32_t latency = llvm::cast<Scheduled>(op.getOperation()).getLatency();
    auto module = rewriter.create<FExtModuleOp>(
        rewriter.getUnknownLoc(),
        mlir::StringAttr::get(rewriter.getContext(), name),
        circt::firrtl::ConventionAttr::get(rewriter.getContext(),
                                           circt::firrtl::Convention::Internal),
        ports,
        StringRef(name));
    module->setAttr(INSTANCE_LATENCY_ATTR,
                    mlir::IntegerAttr::get(attrType, latency));
    return module;
  }

  virtual void
  remapUses(OperationType &oldOp, AdaptorType &adaptor,
            InstanceOp &newOp, Rewriter &rewriter) const {
    using circt::firrtl::utils::getClockVarFromOpBlock;
  
    unsigned id = 1;
    auto adaptorOperands = adaptor.getOperands();
    for (; id <= (newOp.getNumResults() - 2); ++id) {
      rewriter.create<circt::firrtl::ConnectOp>(
          rewriter.getUnknownLoc(),
          newOp->getResult(id),
          adaptorOperands[id - 1]
      );
    }
    rewriter.create<circt::firrtl::ConnectOp>(
        rewriter.getUnknownLoc(),
        newOp->getResult(id),
        getClockVarFromOpBlock(newOp)
    );

    for (auto &operand: llvm::make_early_inc_range(oldOp.getRes().getUses())) {
      (*this->oldTypeMap)[std::make_pair(operand.getOwner(),
                                         operand.getOperandNumber())] =
                                             operand.get().getType();
      operand.set(newOp->getResult(0));
    }
  };

protected:
  FExtModuleOp findOrCreateModule(OperationType &op,
                                  AdaptorType &adaptor,
                                  Rewriter &rewriter) const {
    std::string moduleName = constructModuleName(op, adaptor);
    CircuitOp circuit = op->template getParentOfType<CircuitOp>();

    auto foundModule = circuit.template lookupSymbol<FExtModuleOp>(moduleName);
    if (foundModule) return foundModule;

    auto saved = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(circuit.getBodyBlock());
    FExtModuleOp newModule = createModule(moduleName, op, adaptor, rewriter);
    rewriter.restoreInsertionPoint(saved);
    return newModule;
  }

public:
  // Inherit existing constructor.
  using FIRRTLOpConversionPattern<OperationType>::FIRRTLOpConversionPattern;

  LogicalResult matchAndRewrite(OperationType oldOp, AdaptorType adaptor,
                                Rewriter &rewriter) const override {
    FExtModuleOp module = findOrCreateModule(oldOp, adaptor, rewriter);

    auto newOp =
        rewriter.create<InstanceOp>(oldOp.getLoc(), module, "placeholder");
    remapUses(oldOp, adaptor, newOp, rewriter);
    rewriter.eraseOp(oldOp);
    return mlir::success();
  }

  virtual ~SchedulableOpConversionPattern() = default;
};

#define CAT(FIRST, SECOND) FIRST ## SECOND
#define CAT_E(FIRST, SECOND) CAT(FIRST, SECOND)
#define OP_CLASS(CLASS_PREF) CAT_E(CLASS_PREF, Op)
#define OP_CLASS_ADAPTOR(CLASS_PREF) OP_CLASS(CLASS_PREF)::Adaptor

#define OP_CLASS_CONV_PATTERN(CLASS_PREF) CAT_E(OP_CLASS(CLASS_PREF), ConversionPattern)
#define SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) OP_CLASS(CLASS_PREF), OP_CLASS_ADAPTOR(CLASS_PREF)

#define DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(CLASS_PREF, OP_NAME)                                               \
class OP_CLASS_CONV_PATTERN(CLASS_PREF)                                                                            \
    : public SchedulableOpConversionPattern< SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) > {                            \
                                                                                                                   \
public:                                                                                                            \
  using SchedulableOpConversionPattern<  SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) >::SchedulableOpConversionPattern; \
                                                                                                                   \
  std::string getBaseModuleName() const override {                                                                 \
    return CAT_E(OP_NAME,_MODULE);                                                                                 \
  }                                                                                                                \
};

// AddOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Add, ADD)

// SubOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Sub, SUB)

// MulOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Mul, MUL)

// DivOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Div, DIV)

// AndOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(And, AND)

// OrOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Or, OR)

// XorOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Xor, XOR)

#define DECL_SCHED_UNARY_ARITH_OP_CONV_PATTERN(CLASS_PREF, OP_NAME)                                                \
class OP_CLASS_CONV_PATTERN(CLASS_PREF)                                                                            \
    : public SchedulableOpConversionPattern< SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) > {                            \
                                                                                                                   \
public:                                                                                                            \
  using SchedulableOpConversionPattern<  SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) >::SchedulableOpConversionPattern; \
                                                                                                                   \
  std::string getBaseModuleName() const override {                                                                 \
    return CAT_E(OP_NAME,_MODULE);                                                                                 \
  }                                                                                                                \
};

// NotOpConversionPattern.
DECL_SCHED_UNARY_ARITH_OP_CONV_PATTERN(Not, NOT)

// NegOpConversionPattern.
DECL_SCHED_UNARY_ARITH_OP_CONV_PATTERN(Neg, NEG)

// LessOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Less, LESS)

// LessEqOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(LessEq, LESSEQ)

// GreaterOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Greater, GREATER)

// GreaterEqOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(GreaterEq, GREATEREQ)

// EqOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Eq, EQ)

// NotEqOpConversionPattern.
DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(NotEq, NEQ)

// CastOpConversionPattern.
DECL_SCHED_UNARY_ARITH_OP_CONV_PATTERN(Cast, CAST)

class ShiftLeftOpConversionPattern 
    : public FIRRTLOpConversionPattern<ShiftLeftOp> {
public:
  using FIRRTLOpConversionPattern<ShiftLeftOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ShiftLeftOp::Adaptor;
  using ShlPrimOp = circt::firrtl::ShlPrimOp;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(ShiftLeftOp shLeftOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::getBitWidth;
    using circt::firrtl::SIntType;
    using circt::firrtl::AsSIntPrimOp;
    using circt::firrtl::FIRRTLBaseType;
    using circt::firrtl::BitsPrimOp;
    using circt::firrtl::AsUIntPrimOp;

    auto oldType = getTypeConverter()->convertType(shLeftOp->getResult(0).getType());
    uint32_t oldWidth = *getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(oldType));
    bool isSInt = llvm::isa<SIntType>(oldType);
    Value oldInput = adaptor.getFirst();
    Location oldLoc = shLeftOp.getLoc();

    if (isSInt) {
      auto newReinterpret = rewriter.create<AsUIntPrimOp>(
          oldLoc,
          oldInput
      );

      oldLoc = rewriter.getUnknownLoc();
      oldInput = newReinterpret->getResult(0);
    }
    
    auto newShl = rewriter.create<ShlPrimOp>(
        oldLoc,
        oldInput,
        adaptor.getBits()
    );
    auto newShlResult = newShl->getResult(0);
    auto newShlResultType = newShlResult.getType();
    uint32_t newWidth =
        *getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(newShlResultType));

    Operation *newOp = nullptr;

    newOp = rewriter.create<BitsPrimOp>(
        rewriter.getUnknownLoc(),
        newShlResult,
        newWidth - 1,
        newWidth - oldWidth
    );

    if (isSInt) {
      auto newReinterpret = rewriter.create<AsSIntPrimOp>(
          rewriter.getUnknownLoc(),
          newOp->getResult(0)
      );

      newOp = newReinterpret;
    }

    for (auto &operand:
        llvm::make_early_inc_range(shLeftOp.getRes().getUses())) {
      (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                   operand.getOperandNumber())] =
                                       operand.get().getType();
      operand.set(newOp->getResult(0));
    }

    rewriter.eraseOp(shLeftOp);
    return mlir::success();
  }
};

class ShiftRightOpConversionPattern
    : public FIRRTLOpConversionPattern<ShiftRightOp> {
public:
  using FIRRTLOpConversionPattern<ShiftRightOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ShiftRightOp::Adaptor;
  using ShrPrimOp = circt::firrtl::ShrPrimOp;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(ShiftRightOp shRightOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::getBitWidth;
    using circt::firrtl::SIntType;
    using circt::firrtl::AsSIntPrimOp;
    using circt::firrtl::FIRRTLBaseType;
    using circt::firrtl::UIntType;
    using circt::firrtl::ConstantOp;
    using circt::firrtl::CatPrimOp;
    using circt::firrtl::AsUIntPrimOp;

    auto oldType = getTypeConverter()->convertType(shRightOp->getResult(0).getType());
    uint32_t oldWidth = *getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(oldType));
    bool isSInt = llvm::isa<SIntType>(oldType);
    Value oldInput = adaptor.getFirst();
    Location oldLoc = shRightOp.getLoc();

    if (isSInt) {
      auto newReinterpret = rewriter.create<AsUIntPrimOp>(
          oldLoc,
          oldInput
      );

      oldLoc = rewriter.getUnknownLoc();
      oldInput = newReinterpret->getResult(0);
    }

    auto newShr = rewriter.create<ShrPrimOp>(
        oldLoc,
        oldInput,
        adaptor.getBits()
    );
    auto newShrResult = newShr->getResult(0);
    auto newShrResultType = newShrResult.getType();
    uint32_t newWidth =
        *getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(newShrResultType));
    
    Operation *newOp = nullptr;

    auto newConstType = UIntType::get(
        rewriter.getContext(),
        static_cast<int32_t>(oldWidth - newWidth),
        true // isConst
    );
    auto newConstAttrType = IntegerType::get(
        rewriter.getContext(),
        oldWidth - newWidth,
        IntegerType::SignednessSemantics::Unsigned
    );
    auto newConstAttr = IntegerAttr::get(
        newConstAttrType,
        0 // value
    );
    auto newConst = rewriter.create<ConstantOp>(
        rewriter.getUnknownLoc(),
        newConstType,
        newConstAttr
    );
    auto newConstResult = newConst->getResult(0);

    newOp = rewriter.create<CatPrimOp>(
        rewriter.getUnknownLoc(),
        newConstResult,
        newShrResult
    );

    if (isSInt) {
      auto newReinterpret = rewriter.create<AsSIntPrimOp>(
          rewriter.getUnknownLoc(),
          newOp->getResult(0)
      );

      newOp = newReinterpret;
    }

    for (auto &operand:
        llvm::make_early_inc_range(shRightOp.getRes().getUses())) {
      (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                   operand.getOperandNumber())] =
                                       operand.get().getType();
      operand.set(newOp->getResult(0));
    }

    rewriter.eraseOp(shRightOp);
    return mlir::success();
  }
};

class ConnectOpConversionPattern
    : public FIRRTLOpConversionPattern<ConnectOp> {
public:
  using FIRRTLOpConversionPattern<ConnectOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ConnectOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(ConnectOp connectOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {

    auto newOp = rewriter.create<circt::firrtl::ConnectOp>(
        rewriter.getUnknownLoc(),
        adaptor.getDest(),
        adaptor.getSrc()
    );
    rewriter.replaceOp(connectOp, newOp);
    return mlir::success();
  }
};

class MuxOpConversionPattern : public FIRRTLOpConversionPattern<MuxOp> {
public:
  using FIRRTLOpConversionPattern<MuxOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename MuxOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(MuxOp muxOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::UIntType;
    using circt::firrtl::AsUIntPrimOp;
    using circt::firrtl::MultibitMuxOp;

    Value oldControl = adaptor.getControl();
    Location oldLoc = muxOp.getLoc();

    if (!llvm::isa<UIntType>(oldControl.getType())) {
      auto newReinterpret = rewriter.create<AsUIntPrimOp>(
          oldLoc,
          oldControl
      );

      oldLoc = rewriter.getUnknownLoc();
      oldControl = newReinterpret->getResult(0);
    }

    auto newOp = rewriter.create<MultibitMuxOp>(
        oldLoc,
        oldControl,
        adaptor.getVars());

    for (auto &operand: llvm::make_early_inc_range(muxOp.getRes().getUses())) {
      (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                   operand.getOperandNumber())] =
                                       operand.get().getType();
      operand.set(newOp.getResult());
    }
    rewriter.eraseOp(muxOp);
    return mlir::success();
  }
};

class LatencyOpConversionPattern
    : public SchedulableOpConversionPattern<LatencyOp, LatencyOp::Adaptor> {

public:
  using SchedulableOpConversionPattern<LatencyOp, LatencyOp::Adaptor>::SchedulableOpConversionPattern;

  std::string
  getBaseModuleName() const override {
    return BUF_MODULE;
  }

  std::string
  constructModuleName(LatencyOp &op, LatencyOp::Adaptor &adaptor) const override {
    using circt::firrtl::getBitWidth;
    using circt::firrtl::FIRRTLBaseType;

    std::string name = getBaseModuleName();
    llvm::raw_string_ostream nameStream(name);
  
    auto convertedType = getTypeConverter()->convertType(op->getResult(0).getType());
    auto width = getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(convertedType));
    int32_t latency = llvm::cast<Scheduled>(op.getOperation()).getLatency();
    nameStream << "_IN_" << *width << "_OUT_" << *width << "_" << latency;

    return name;
  }
};

class DFCIRToFIRRTLPass
    : public impl::DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass> {
public:
  using ConvertedOps = mlir::DenseSet<mlir::Operation *>;
  using OldTypeMap =
      std::unordered_map<std::pair<mlir::Operation *, unsigned>, mlir::Type>;

  explicit DFCIRToFIRRTLPass() : impl::DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass>() {}

  void runOnOperation() override {
    // Define the conversion target.
    ConversionTarget target(getContext());
    target.addLegalDialect<DFCIRDialect>();
    target.addIllegalOp<KernelOp>();
    target.addLegalDialect<circt::firrtl::FIRRTLDialect>();

    // TODO: Implement 'FIRRTLTypeConverter' completely.
    // Issue #12 (https://github.com/ispras/utopia-hls/issues/12).
    FIRRTLTypeConverter typeConverter;
    ConvertedOps convertedOps;
    OldTypeMap oldTypeMap;

    ModuleArgMap moduleArgMap;

    // Convert the kernel first to get a FIRRTL-circuit.
    RewritePatternSet patterns(&getContext());

    patterns.add<KernelOpConversionPattern>(
        &getContext(),
        typeConverter,
        &convertedOps,
        &oldTypeMap,
        &moduleArgMap
    );

    // Apply partial conversion.
    if (failed(applyPartialConversion(getOperation(),
                                      target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }

    // Define the rest of the rewrite patterns.
    patterns.clear();
    target.addIllegalDialect<DFCIRDialect>();
    target.addIllegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<OffsetOp>();
    patterns.add<
        InputOpConversionPattern,
        ScalarInputOpConversionPattern,
        OutputOpConversionPattern,
        ScalarOutputOpConversionPattern,
        ConstantOpConversionPattern,
        MuxOpConversionPattern,
        AddOpConversionPattern,
        SubOpConversionPattern,
        MulOpConversionPattern,
        DivOpConversionPattern,
        AndOpConversionPattern,
        OrOpConversionPattern,
        XorOpConversionPattern,
        NotOpConversionPattern,
        NegOpConversionPattern,
        LessOpConversionPattern,
        LessEqOpConversionPattern,
        GreaterOpConversionPattern,
        GreaterEqOpConversionPattern,
        EqOpConversionPattern,
        NotEqOpConversionPattern,
        CastOpConversionPattern,
        ShiftLeftOpConversionPattern,
        ShiftRightOpConversionPattern,
        ConnectOpConversionPattern,
        LatencyOpConversionPattern>(
        &getContext(),
        typeConverter,
        &convertedOps,
        &oldTypeMap,
        &moduleArgMap
    );

    // Apply partial conversion.
    if (failed(applyPartialConversion(getOperation(),
                                      target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createDFCIRToFIRRTLPass() {
  return std::make_unique<DFCIRToFIRRTLPass>();
}

} // namespace mlir::dfcir