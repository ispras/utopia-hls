//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "dfcir/passes/ModuleDefines.inc"

namespace circt::firrtl::utils {

Value getBlockArgument(Block *block, unsigned ind) {
  return block->getArgument(ind);
}

Value getBlockArgumentFromOpBlock(Operation *op, unsigned ind) {
  return getBlockArgument(op->getBlock(), ind);
}

Value getClockVar(Block *block) {
  Value arg = getBlockArgument(block, block->getNumArguments() - 1);
  if (mlir::isa<ClockType>(arg.getType())) {
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

#include "dfcir/passes/DFCIRPasses.h.inc"

class FIRRTLTypeConverter : public TypeConverter {
public:
  FIRRTLTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](DFCIRRawBitsType type) -> Type {
      return circt::firrtl::UIntType::get(type.getContext(), type.getBits());
    });
    addConversion([](DFCIRFixedType type) -> Type {
      uint32_t width = uint32_t(type.getSign()) +
                       type.getIntegerBits() +
                       type.getFractionBits();
      if (type.getSign()) {
        return circt::firrtl::SIntType::get(type.getContext(), width);
      } else {
        return circt::firrtl::UIntType::get(type.getContext(), width);
      }
    });
    addConversion([](DFCIRFloatType type) -> Type {
      uint32_t width = 1 + type.getExponentBits() + type.getFractionBits();
      return circt::firrtl::UIntType::get(type.getContext(), width);
    });
    addConversion([this](DFCIRStreamType type) -> Type {
      return convertType(type.getStreamType());
    });
    addConversion([this](DFCIRScalarType type) -> Type {
      return convertType(type.getScalarType());
    });
    addConversion([this](DFCIRConstantType type) -> Type {
      return convertType(type.getConstType());
    });
  }
};

typedef std::unordered_map<mlir::Operation *, unsigned> ModuleArgMap;

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

    nameStream << "_"
               << llvm::cast<Scheduled>(op.getOperation()).getPosLatency();

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

    int32_t latency = llvm::cast<Scheduled>(op.getOperation()).getPosLatency();
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
class AndOpConversionPattern
    : public FIRRTLOpConversionPattern<AndOp> {
public:
  using FIRRTLOpConversionPattern<AndOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename AndOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(AndOp andOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::SIntType;
    using circt::firrtl::AsSIntPrimOp;

    auto newType = getTypeConverter()->convertType(andOp->getResult(0).getType());
    Operation *newOp = rewriter.create<circt::firrtl::AndPrimOp>(
        rewriter.getUnknownLoc(),
        newType,
        adaptor.getFirst(),
        adaptor.getSecond()
    );

    if (llvm::isa<SIntType>(newType)) {
      newOp = rewriter.create<AsSIntPrimOp>(
        rewriter.getUnknownLoc(),
        newOp->getResult(0)
      );
    }

    for (auto &operand:
      llvm::make_early_inc_range(andOp.getRes().getUses())) {
    (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                 operand.getOperandNumber())] =
                                     operand.get().getType();
    operand.set(newOp->getResult(0));
  }

  rewriter.eraseOp(andOp);
  return mlir::success();
  }
};

// OrOpConversionPattern.
class OrOpConversionPattern
    : public FIRRTLOpConversionPattern<OrOp> {
public:
  using FIRRTLOpConversionPattern<OrOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename OrOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(OrOp orOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::SIntType;
    using circt::firrtl::AsSIntPrimOp;

    auto newType = getTypeConverter()->convertType(orOp->getResult(0).getType());
    Operation *newOp = rewriter.create<circt::firrtl::OrPrimOp>(
        rewriter.getUnknownLoc(),
        newType,
        adaptor.getFirst(),
        adaptor.getSecond()
    );

    if (llvm::isa<SIntType>(newType)) {
      newOp = rewriter.create<AsSIntPrimOp>(
        rewriter.getUnknownLoc(),
        newOp->getResult(0)
      );
    }

    for (auto &operand:
      llvm::make_early_inc_range(orOp.getRes().getUses())) {
    (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                 operand.getOperandNumber())] =
                                     operand.get().getType();
    operand.set(newOp->getResult(0));
  }

  rewriter.eraseOp(orOp);
  return mlir::success();
  }
};

// XorOpConversionPattern.
class XorOpConversionPattern
    : public FIRRTLOpConversionPattern<XorOp> {
public:
  using FIRRTLOpConversionPattern<XorOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename XorOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(XorOp xorOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::SIntType;
    using circt::firrtl::AsSIntPrimOp;

    auto newType = getTypeConverter()->convertType(xorOp->getResult(0).getType());
    Operation *newOp = rewriter.create<circt::firrtl::XorPrimOp>(
        rewriter.getUnknownLoc(),
        newType,
        adaptor.getFirst(),
        adaptor.getSecond()
    );

    if (llvm::isa<SIntType>(newType)) {
      newOp = rewriter.create<AsSIntPrimOp>(
        rewriter.getUnknownLoc(),
        newOp->getResult(0)
      );
    }

    for (auto &operand:
      llvm::make_early_inc_range(xorOp.getRes().getUses())) {
    (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                 operand.getOperandNumber())] =
                                     operand.get().getType();
    operand.set(newOp->getResult(0));
  }

  rewriter.eraseOp(xorOp);
  return mlir::success();
  }
};

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
class NotOpConversionPattern
    : public FIRRTLOpConversionPattern<NotOp> {
public:
  using FIRRTLOpConversionPattern<NotOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename NotOp::Adaptor;
  using Rewriter = ConversionPatternRewriter;

  LogicalResult matchAndRewrite(NotOp notOp, OpAdaptor adaptor,
                                Rewriter &rewriter) const override {
    using circt::firrtl::SIntType;
    using circt::firrtl::AsSIntPrimOp;

    auto newType = getTypeConverter()->convertType(notOp->getResult(0).getType());
    Operation *newOp = rewriter.create<circt::firrtl::NotPrimOp>(
        rewriter.getUnknownLoc(),
        newType,
        adaptor.getFirst()
    );

    if (llvm::isa<SIntType>(newType)) {
      newOp = rewriter.create<AsSIntPrimOp>(
        rewriter.getUnknownLoc(),
        newOp->getResult(0)
      );
    }

    for (auto &operand:
      llvm::make_early_inc_range(notOp.getRes().getUses())) {
    (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                 operand.getOperandNumber())] =
                                     operand.get().getType();
    operand.set(newOp->getResult(0));
  }

  rewriter.eraseOp(notOp);
  return mlir::success();
  }
};

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
    using circt::firrtl::FIRRTLBaseType;
    using circt::firrtl::BitsPrimOp;
    using circt::firrtl::CatPrimOp;
    using circt::firrtl::AsSIntPrimOp;

    auto oldType = getTypeConverter()->convertType(shLeftOp->getResult(0).getType());
    uint32_t oldWidth = *getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(oldType));
    bool isSInt = llvm::isa<SIntType>(oldType);

    Operation *newOp = nullptr;

    auto newShl = rewriter.create<ShlPrimOp>(
      shLeftOp.getLoc(),
      adaptor.getFirst(),
      adaptor.getBits()
    );

    if (isSInt) {
      auto getSignOp = rewriter.create<BitsPrimOp>(
        rewriter.getUnknownLoc(),
        adaptor.getFirst(),
        oldWidth - 1,
        oldWidth - 1
      );

      auto castedSignOp = rewriter.create<AsSIntPrimOp>(
        rewriter.getUnknownLoc(),
        getSignOp->getResult(0)
      );

      auto bitsOp = rewriter.create<BitsPrimOp>(
        rewriter.getUnknownLoc(),
        newShl->getResult(0),
        oldWidth - 2,
        0
      );

      auto castedBitsOp = rewriter.create<AsSIntPrimOp>(
        rewriter.getUnknownLoc(),
        bitsOp->getResult(0)
      );

      auto catOp = rewriter.create<CatPrimOp>(
        rewriter.getUnknownLoc(),
        castedSignOp->getResult(0),
        castedBitsOp->getResult(0)
      );

      newOp = rewriter.create<AsSIntPrimOp>(
        rewriter.getUnknownLoc(),
        catOp->getResult(0)
      );

    } else {
      newOp = rewriter.create<BitsPrimOp>(
        rewriter.getUnknownLoc(),
        newShl->getResult(0),
        oldWidth - 1,
        0
      );
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
    using circt::firrtl::FIRRTLBaseType;
    using circt::firrtl::PadPrimOp;

    auto oldType = getTypeConverter()->convertType(shRightOp->getResult(0).getType());
    uint32_t oldWidth = *getBitWidth(llvm::dyn_cast<FIRRTLBaseType>(oldType));

    auto newShr = rewriter.create<ShrPrimOp>(
      shRightOp.getLoc(),
      adaptor.getFirst(),
      adaptor.getBits()
    );

    auto oldWidthAttrType = IntegerType::get(
      rewriter.getContext(),
      32, // width
      IntegerType::SignednessSemantics::Signless
    );

    auto oldWidthAttr = IntegerAttr::get(
      oldWidthAttrType,
      oldWidth // value
    );

    // Sign- or zero-extend the previously
    // shifted value to restore the original bit width.
    auto newOp = rewriter.create<PadPrimOp>(
      rewriter.getUnknownLoc(),
      oldType,
      newShr->getResult(0),
      oldWidthAttr
    );

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

class BitsOpConversionPattern : public FIRRTLOpConversionPattern<BitsOp> {
  public:
    using FIRRTLOpConversionPattern<BitsOp>::FIRRTLOpConversionPattern;
    using OpAdaptor = typename BitsOp::Adaptor;
    using Rewriter = ConversionPatternRewriter;

    LogicalResult matchAndRewrite(BitsOp bitsOp, OpAdaptor adaptor,
                                  Rewriter &rewriter) const override {
      using circt::firrtl::BitsPrimOp;

      auto leftAttr = adaptor.getLeft();
      auto rightAttr = adaptor.getRight();

      if (leftAttr.getSInt() < rightAttr.getSInt()) {
        IntegerAttr buf;
        buf = leftAttr;
        rightAttr = leftAttr;
        leftAttr = buf;
      }

      auto newOp = rewriter.create<BitsPrimOp>(
          bitsOp.getLoc(),
          adaptor.getInput(),
          leftAttr,
          rightAttr
      );

      for (auto &operand: llvm::make_early_inc_range(bitsOp.getRes().getUses())) {
        (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                     operand.getOperandNumber())] =
                                         operand.get().getType();
        operand.set(newOp->getResult(0));
      }
      rewriter.eraseOp(bitsOp);
      return mlir::success();
    }
};

class CatOpConversionPattern : public FIRRTLOpConversionPattern<CatOp> {
  public:
    using FIRRTLOpConversionPattern<CatOp>::FIRRTLOpConversionPattern;
    using OpAdaptor = typename CatOp::Adaptor;
    using Rewriter = ConversionPatternRewriter;

    LogicalResult matchAndRewrite(CatOp catOp, OpAdaptor adaptor,
                                  Rewriter &rewriter) const override {
      using circt::firrtl::CatPrimOp;

      auto newOp = rewriter.create<CatPrimOp>(
          catOp.getLoc(),
          adaptor.getFirst(),
          adaptor.getSecond()
      );

      for (auto &operand: llvm::make_early_inc_range(catOp.getRes().getUses())) {
        (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                     operand.getOperandNumber())] =
                                         operand.get().getType();
        operand.set(newOp->getResult(0));
      }
      rewriter.eraseOp(catOp);
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
    int32_t latency = llvm::cast<Scheduled>(op.getOperation()).getPosLatency();
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
        BitsOpConversionPattern,
        CatOpConversionPattern,
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
