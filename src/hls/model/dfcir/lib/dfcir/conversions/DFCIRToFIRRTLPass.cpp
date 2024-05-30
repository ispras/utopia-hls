#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

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
  using OffsetMap = std::unordered_map<
          std::pair<mlir::Operation *, unsigned>, signed>;
  using OldTypeMap = std::unordered_map<
          std::pair<mlir::Operation *, unsigned>, mlir::Type>;

  mutable ConvertedOps *convertedOps;
  const LatencyConfig *latencyConfig;
  OffsetMap *offsetMap;
  OldTypeMap *oldTypeMap;   // TODO: Change when replaceAllUsesWith-related pull request for MLIR is approved.
  ModuleArgMap *moduleArgMap;

  FIRRTLOpConversionPattern(MLIRContext *context,
                            TypeConverter &typeConverter,
                            ConvertedOps *convertedOps,
                            LatencyConfig *latencyConfig,
                            OffsetMap *offsetMap,
                            OldTypeMap *oldTypeMap,
                            ModuleArgMap *moduleArgMap)
          : OpConversionPattern<OperationType>(typeConverter, context),
            convertedOps(convertedOps),
            latencyConfig(latencyConfig),
            offsetMap(offsetMap),
            oldTypeMap(oldTypeMap),
            moduleArgMap(moduleArgMap) {
    // Required to allow root updates, which imply recursive
    // pattern application.
    //Pattern::setHasBoundedRewriteRecursion(true);
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

  LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Block *kernelBlock = &(kernelOp.getBodyRegion().getBlocks().front());
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
        Type converted = getTypeConverter()->convertType(
                op.getResult(0).getType());
        modulePorts.emplace_back(
                mlir::StringAttr::get(getContext(), name),
                converted,
                (llvm::isa<InputOp, ScalarInputOp>(op))
                ? circt::firrtl::Direction::In :
                circt::firrtl::Direction::Out);
      }
    }

    // Add explicit clock argument.

    modulePorts.emplace_back(
            mlir::StringAttr::get(rewriter.getContext(), CLOCK_ARG),
            circt::firrtl::ClockType::get(rewriter.getContext()),
            circt::firrtl::Direction::In);

    // Add a module to represent the old kernel with.
    rewriter.setInsertionPointToStart(circuitOp.getBodyBlock());
    auto fModuleOp = rewriter.create<FModuleOp>(
            rewriter.getUnknownLoc(),
            StringAttr::get(rewriter.getContext(), kernelOp.getName()),
            ConventionAttr::get(rewriter.getContext(),
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
    rewriter.mergeBlocks(kernelBlock,
                         fModuleOp.getBodyBlock()
            //,fModuleOp.getBodyBlock()->getArguments()
    );
    rewriter.restoreInsertionPoint(save);
    rewriter.replaceOp(kernelOp, circuitOp);

    return mlir::success();
  }
};

class InputOpConversionPattern : public FIRRTLOpConversionPattern<InputOp> {
public:
  using FIRRTLOpConversionPattern<InputOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename InputOp::Adaptor;

  LogicalResult matchAndRewrite(InputOp inputOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // TODO: Add control stream functionality.
    rewriter.eraseOp(inputOp);
    return mlir::success();
  }
};

class ScalarInputOpConversionPattern
        : public FIRRTLOpConversionPattern<ScalarInputOp> {
public:
  using FIRRTLOpConversionPattern<ScalarInputOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ScalarInputOp::Adaptor;

  LogicalResult matchAndRewrite(ScalarInputOp scalarInputOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(scalarInputOp);
    return mlir::success();
  }
};

class OutputOpConversionPattern : public FIRRTLOpConversionPattern<OutputOp> {
public:
  using FIRRTLOpConversionPattern<OutputOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename OutputOp::Adaptor;

  LogicalResult matchAndRewrite(OutputOp outputOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    using circt::firrtl::utils::createConnect;
    using circt::firrtl::utils::getBlockArgumentFromOpBlock;

    // TODO: Add control stream functionality.
    if (outputOp.getStream()) {
      createConnect(rewriter,
                    getBlockArgumentFromOpBlock(outputOp,
                                                (*moduleArgMap)[outputOp]),
                    adaptor.getStream());
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

  LogicalResult
  matchAndRewrite(ScalarOutputOp scalarOutputOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using circt::firrtl::utils::createConnect;
    using circt::firrtl::utils::getBlockArgumentFromOpBlock;

    // TODO: Add control stream functionality.
    if (scalarOutputOp.getStream()) {
      createConnect(rewriter,
                    getBlockArgumentFromOpBlock(scalarOutputOp,
                                                (*moduleArgMap)[scalarOutputOp]),
                    adaptor.getStream());
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

  LogicalResult matchAndRewrite(ConstantOp constOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
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

#define CAT(FIRST, SECOND) FIRST ## SECOND
#define CAT_E(FIRST, SECOND) CAT(FIRST, SECOND)

#define GET_SCHED_OP_NAME(OP_NAME)                                           \
std::string name = CAT_E(OP_NAME,_MODULE);                                   \
llvm::raw_string_ostream nameStream(name);                                   \
                                                                             \
for (unsigned id = 0; id < op->getNumOperands(); ++id) {                     \
  nameStream << "_IN_";                                                      \
  Type oldType = (*oldTypeMap)[std::make_pair(op, id)];                      \
  Type innerType = llvm::cast<DFType>(oldType).getDFType();                  \
  llvm::cast<SVSynthesisable>(innerType).printSVSignature(nameStream);       \
}                                                                            \
                                                                             \
for (unsigned id = 0; id < op->getNumResults(); ++id) {                      \
  nameStream << "_OUT_";                                                     \
  Type oldType = (*oldTypeMap)[std::make_pair(op, id)];                      \
  Type innerType = llvm::cast<DFType>(oldType).getDFType();                  \
  llvm::cast<SVSynthesisable>(innerType).printSVSignature(nameStream);       \
  bool isFloat = innerType.isa<DFCIRFloatType>();                            \
  if (isFloat) {                                                             \
    nameStream << "_" << latencyConfig->find(CAT_E(OP_NAME,_FLOAT))->second; \
  } else {                                                                   \
    nameStream << "_" << latencyConfig->find(CAT_E(OP_NAME,_INT))->second;   \
  }                                                                          \
}                                                                            \
return name;

#define GET_OP_SV_PARAMS(CTX, ATTR_TYPE, LATENCY, WIDTH) {      \
//circt::firrtl::ParamDeclAttr::get(                              \
//        CTX,                                                    \
//        mlir::StringAttr::get(                                  \
//        CTX,                                                    \
//        STAGES_PARAM),                                          \
//        ATTR_TYPE,                                              \
//        mlir::IntegerAttr::get(                                 \
//        ATTR_TYPE, LATENCY)),                                   \
//circt::firrtl::ParamDeclAttr::get(                              \
//        CTX,                                                    \
//        mlir::StringAttr::get(                                  \
//        CTX,                                                    \
//        "op_" TYPE_SIZE_PARAM),                                 \
//        ATTR_TYPE,                                              \
//        mlir::IntegerAttr::get(                                 \
//        ATTR_TYPE,                                              \
//        WIDTH))                                                 \
}

template <typename OperationType, typename AdaptorType>
class SchedulableOpConversionPattern {
  using FExtModuleOp = circt::firrtl::FExtModuleOp;
  using InstanceOp = circt::firrtl::InstanceOp;
  using CircuitOp = circt::firrtl::CircuitOp;


  virtual std::string
  constructModuleName(const OperationType &op, AdaptorType &adaptor) const = 0;

  virtual FExtModuleOp
  createModule(const std::string &name, const OperationType &op,
               AdaptorType &adaptor,
               ConversionPatternRewriter &rewriter) const = 0;

  virtual void
  remapUses(OperationType &oldOp, AdaptorType &adaptor, InstanceOp &newOp,
            ConversionPatternRewriter &rewriter) const = 0;

protected:
  FExtModuleOp findOrCreateModule(const OperationType &op, AdaptorType &adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    std::string moduleName = constructModuleName(op, adaptor);
    CircuitOp circuit = circt::firrtl::utils::findCircuit(op);

    auto foundModule = circuit.template lookupSymbol<FExtModuleOp>(moduleName);
    if (foundModule) return foundModule;

    auto saved = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(circuit.getBodyBlock());
    FExtModuleOp newModule = createModule(moduleName, op, adaptor, rewriter);
    rewriter.restoreInsertionPoint(saved);
    return newModule;
  }

  virtual ~SchedulableOpConversionPattern() = default;
};

#define OP_CLASS(CLASS_PREF) CAT_E(CLASS_PREF, Op)
#define OP_CLASS_ADAPTOR(CLASS_REF) OP_CLASS(CLASS_REF)::Adaptor

#define OP_CLASS_CONV_PATTERN(CLASS_PREF) CAT_E(OP_CLASS(CLASS_PREF),ConversionPattern)
#define SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) OP_CLASS(CLASS_PREF) , OP_CLASS_ADAPTOR(CLASS_PREF)

#define DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(CLASS_PREF, OP_NAME)                                \
class OP_CLASS_CONV_PATTERN(CLASS_PREF) : public FIRRTLOpConversionPattern< OP_CLASS(CLASS_PREF) >, \
           SchedulableOpConversionPattern< SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) > {               \
public:                                                                                             \
  using FIRRTLOpConversionPattern<OP_CLASS(CLASS_PREF)>::FIRRTLOpConversionPattern;                 \
  using OpAdaptor = typename OP_CLASS_ADAPTOR(CLASS_PREF);                                          \
  using FExtModuleOp = circt::firrtl::FExtModuleOp;                                                 \
  using InstanceOp = circt::firrtl::InstanceOp;                                                     \
  using IntType = circt::firrtl::IntType;                                                           \
                                                                                                    \
  std::string                                                                                       \
  constructModuleName(const OP_CLASS(CLASS_PREF) &op, OpAdaptor &adaptor) const override {          \
    GET_SCHED_OP_NAME(OP_NAME)                                                                      \
  }                                                                                                 \
                                                                                                    \
  FExtModuleOp                                                                                      \
  createModule(const std::string &name, const OP_CLASS(CLASS_PREF) &op, OpAdaptor &adaptor,         \
               ConversionPatternRewriter &rewriter) const override {                                \
    Type type = op->getResult(0).getType();                                                         \
    Type innerType = llvm::cast<DFType>(type).getDFType();                                          \
    Type firstType = (*oldTypeMap)[std::make_pair(op, 0)];                                          \
    Type firstInnerType = llvm::cast<DFType>(firstType).getDFType();                                \
    Type secondType = (*oldTypeMap)[std::make_pair(op, 1)];                                         \
    Type secondInnerType = llvm::cast<DFType>(secondType).getDFType();                              \
    Type convertedType = getTypeConverter()->convertType(type);                                     \
    auto ports = getBinaryOpPorts(convertedType, adaptor.getFirst().getType(),                      \
                                  adaptor.getSecond().getType(),                                    \
                                  rewriter.getContext());                                           \
    IntegerType attrType = mlir::IntegerType::get(rewriter.getContext(), 32,                        \
                                                  mlir::IntegerType::Unsigned);                     \
    auto outTypeWidth = llvm::cast<SVSynthesisable>(innerType).getBitWidth();                       \
    auto firstTypeWidth = llvm::cast<SVSynthesisable>(firstInnerType).getBitWidth();                \
    auto secondTypeWidth = llvm::cast<SVSynthesisable>(secondInnerType).getBitWidth();              \
    assert(outTypeWidth == firstTypeWidth && outTypeWidth == secondTypeWidth);                      \
                                                                                                    \
    bool isFloat = innerType.isa<DFCIRFloatType>();                                                 \
    unsigned latency = latencyConfig->find(                                                         \
            (isFloat) ? CAT_E(OP_NAME,_FLOAT) : CAT_E(OP_NAME,_INT))->second;                       \
    auto module = rewriter.create<FExtModuleOp>(                                                    \
            rewriter.getUnknownLoc(),                                                               \
            mlir::StringAttr::get(rewriter.getContext(), name),                                     \
            circt::firrtl::ConventionAttr::get(rewriter.getContext(),                               \
                                               circt::firrtl::Convention::Internal),                \
            ports,                                                                                  \
            StringRef(name));                                                                      \
    module->setAttr(INSTANCE_LATENCY_ATTR,                                                          \
                    mlir::IntegerAttr::get(attrType, latency));                                     \
    return module;                                                                                  \
  }                                                                                                 \
                                                                                                    \
  void remapUses( OP_CLASS(CLASS_PREF) &oldOp, OpAdaptor &adaptor,                                  \
                 InstanceOp &newOp,                                                                 \
                 ConversionPatternRewriter &rewriter) const override {                              \
    using circt::firrtl::utils::createConnect;                                                      \
    using circt::firrtl::utils::getClockVarFromOpBlock;                                             \
    createConnect(rewriter, newOp.getResult(1), adaptor.getFirst(),                                 \
                  (*offsetMap)[std::make_pair(oldOp, 0)]);                                          \
    createConnect(rewriter, newOp.getResult(2), adaptor.getSecond(),                                \
                  (*offsetMap)[std::make_pair(oldOp, 1)]);                                          \
    createConnect(rewriter, newOp.getResult(3), getClockVarFromOpBlock(newOp));                     \
                                                                                                    \
    for (auto &operand: llvm::make_early_inc_range(oldOp.getRes().getUses())) {                     \
      (*oldTypeMap)[std::make_pair(operand.getOwner(),                                              \
                                   operand.getOperandNumber())] =                                   \
              operand.get().getType();                                                              \
      operand.set(newOp.getResult(0));                                                              \
    }                                                                                               \
  }                                                                                                 \
                                                                                                    \
                                                                                                    \
  LogicalResult matchAndRewrite( OP_CLASS(CLASS_PREF) oldOp, OpAdaptor adaptor,                     \
                                ConversionPatternRewriter &rewriter) const override {               \
    FExtModuleOp module = findOrCreateModule(oldOp, adaptor, rewriter);                             \
                                                                                                    \
    auto newOp = rewriter.create<InstanceOp>(                                                       \
            oldOp.getLoc(),                                                                         \
            module,                                                                                 \
            "placeholder");                                                                         \
    remapUses(oldOp, adaptor, newOp, rewriter);                                                     \
    rewriter.eraseOp(oldOp);                                                                        \
                                                                                                    \
    return mlir::success();                                                                         \
  }                                                                                                 \
};                                                                                                  \

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Add, ADD) // AddOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Sub, SUB) // SubOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Mul, MUL) // MulOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Div, DIV) // DivOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(And, AND) // AndOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Or, OR)   // OrOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Xor, XOR)   // OrOpConversionPattern.

#define DECL_SCHED_UNARY_ARITH_OP_CONV_PATTERN(CLASS_PREF, OP_NAME)                                 \
class OP_CLASS_CONV_PATTERN(CLASS_PREF) : public FIRRTLOpConversionPattern< OP_CLASS(CLASS_PREF) >, \
           SchedulableOpConversionPattern< SCHED_OP_CONV_PATTERN_SPEC(CLASS_PREF) > {               \
public:                                                                                             \
  using FIRRTLOpConversionPattern<OP_CLASS(CLASS_PREF)>::FIRRTLOpConversionPattern;                 \
  using OpAdaptor = typename OP_CLASS_ADAPTOR(CLASS_PREF);                                          \
  using FExtModuleOp = circt::firrtl::FExtModuleOp;                                                 \
  using InstanceOp = circt::firrtl::InstanceOp;                                                     \
  using IntType = circt::firrtl::IntType;                                                           \
                                                                                                    \
  std::string                                                                                       \
  constructModuleName(const OP_CLASS(CLASS_PREF) &op, OpAdaptor &adaptor) const override {          \
    GET_SCHED_OP_NAME(OP_NAME)                                                                      \
  }                                                                                                 \
                                                                                                    \
  FExtModuleOp                                                                                      \
  createModule(const std::string &name, const OP_CLASS(CLASS_PREF) &op, OpAdaptor &adaptor,         \
               ConversionPatternRewriter &rewriter) const override {                                \
    Type type = op->getResult(0).getType();                                                         \
    Type innerType = llvm::cast<DFType>(type).getDFType();                                          \
    Type firstType = (*oldTypeMap)[std::make_pair(op, 0)];                                          \
    Type firstInnerType = llvm::cast<DFType>(firstType).getDFType();                                \
    Type convertedType = getTypeConverter()->convertType(type);                                     \
    auto ports = getUnaryOpPorts(convertedType, adaptor.getFirst().getType(),                       \
                                  rewriter.getContext());                                           \
    IntegerType attrType = mlir::IntegerType::get(rewriter.getContext(), 32,                        \
                                                  mlir::IntegerType::Unsigned);                     \
    auto outTypeWidth = llvm::cast<SVSynthesisable>(innerType).getBitWidth();                       \
    auto firstTypeWidth = llvm::cast<SVSynthesisable>(firstInnerType).getBitWidth();                \
    assert(outTypeWidth == firstTypeWidth);                                                         \
                                                                                                    \
    bool isFloat = innerType.isa<DFCIRFloatType>();                                                 \
    unsigned latency = latencyConfig->find(                                                         \
            (isFloat) ? CAT_E(OP_NAME,_FLOAT) : CAT_E(OP_NAME,_INT))->second;                       \
    auto module = rewriter.create<FExtModuleOp>(                                                    \
            rewriter.getUnknownLoc(),                                                               \
            mlir::StringAttr::get(rewriter.getContext(), name),                                     \
            circt::firrtl::ConventionAttr::get(rewriter.getContext(),                               \
                                               circt::firrtl::Convention::Internal),                \
            ports,                                                                                  \
            StringRef(name));                                                                     \
    module->setAttr(INSTANCE_LATENCY_ATTR,                                                          \
                    mlir::IntegerAttr::get(attrType, latency));                                     \
    return module;                                                                                  \
  }                                                                                                 \
                                                                                                    \
  void remapUses( OP_CLASS(CLASS_PREF) &oldOp, OpAdaptor &adaptor,                                  \
                 InstanceOp &newOp,                                                                 \
                 ConversionPatternRewriter &rewriter) const override {                              \
    using circt::firrtl::utils::createConnect;                                                      \
    using circt::firrtl::utils::getClockVarFromOpBlock;                                             \
    createConnect(rewriter, newOp.getResult(1), adaptor.getFirst(),                                 \
                  (*offsetMap)[std::make_pair(oldOp, 0)]);                                          \
    createConnect(rewriter, newOp.getResult(2), getClockVarFromOpBlock(newOp));                     \
                                                                                                    \
    for (auto &operand: llvm::make_early_inc_range(oldOp.getRes().getUses())) {                     \
      (*oldTypeMap)[std::make_pair(operand.getOwner(),                                              \
                                   operand.getOperandNumber())] =                                   \
              operand.get().getType();                                                              \
      operand.set(newOp.getResult(0));                                                              \
    }                                                                                               \
  }                                                                                                 \
                                                                                                    \
                                                                                                    \
  LogicalResult matchAndRewrite( OP_CLASS(CLASS_PREF) oldOp, OpAdaptor adaptor,                     \
                                ConversionPatternRewriter &rewriter) const override {               \
    FExtModuleOp module = findOrCreateModule(oldOp, adaptor, rewriter);                             \
                                                                                                    \
    auto newOp = rewriter.create<InstanceOp>(                                                       \
            oldOp.getLoc(),                                                                         \
            module,                                                                                 \
            "placeholder");                                                                         \
    remapUses(oldOp, adaptor, newOp, rewriter);                                                     \
    rewriter.eraseOp(oldOp);                                                                        \
                                                                                                    \
    return mlir::success();                                                                         \
  }                                                                                                 \
};

DECL_SCHED_UNARY_ARITH_OP_CONV_PATTERN(Not, NOT)   // NotOpConversionPattern.

DECL_SCHED_UNARY_ARITH_OP_CONV_PATTERN(Neg, NEG)   // NegOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Less, LESS) // LessOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(LessEq, LESS_EQ) // LessEqOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(More, MORE) // MoreOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(MoreEq, MORE_EQ) // MoreEqOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(Eq, EQ) // EqOpConversionPattern.

DECL_SCHED_BINARY_ARITH_OP_CONV_PATTERN(NotEq, NEQ) // NotEqOpConversionPattern.

class ConnectOpConversionPattern : public FIRRTLOpConversionPattern<ConnectOp> {
public:
  using FIRRTLOpConversionPattern<ConnectOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ConnectOp::Adaptor;

  LogicalResult matchAndRewrite(ConnectOp connectOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    using circt::firrtl::utils::createConnect;

    auto newOp = createConnect(rewriter,
                               adaptor.getConnecting(),
                               adaptor.getConnectee());
    rewriter.replaceOp(connectOp, newOp);
    return mlir::success();
  }
};

class OffsetOpConversionPattern : public FIRRTLOpConversionPattern<OffsetOp> {
public:
  using FIRRTLOpConversionPattern<OffsetOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename OffsetOp::Adaptor;

  LogicalResult matchAndRewrite(OffsetOp offsetOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    int offset = adaptor.getOffset().getInt();

    for (auto &operand: llvm::make_early_inc_range(
            offsetOp.getRes().getUses())) {
      (*oldTypeMap)[std::make_pair(operand.getOwner(),
                                   operand.getOperandNumber())] =
              operand.get().getType();
      operand.set(offsetOp.getOperand());
      (*offsetMap)[std::make_pair(operand.getOwner(),
                                  operand.getOperandNumber())] = offset;
    }

    rewriter.eraseOp(offsetOp);
    return mlir::success();
  }
};

class MuxOpConversionPattern : public FIRRTLOpConversionPattern<MuxOp> {
public:
  using FIRRTLOpConversionPattern<MuxOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename MuxOp::Adaptor;

  LogicalResult matchAndRewrite(MuxOp muxOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.create<circt::firrtl::MultibitMuxOp>(
            rewriter.getUnknownLoc(), adaptor.getControl(), adaptor.getVars());
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

class DFCIRToFIRRTLPass
        : public impl::DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass> {
public:
  using ConvertedOps = mlir::DenseSet<mlir::Operation *>;
  using OffsetMap = std::unordered_map<std::pair<mlir::Operation *, unsigned>, signed>;
  using OldTypeMap = std::unordered_map<std::pair<mlir::Operation *, unsigned>, mlir::Type>;

  explicit DFCIRToFIRRTLPass(const DFCIRToFIRRTLPassOptions &options)
          : impl::DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass>(options) {}

  void runOnOperation() override {
    // Define the conversion target.
    ConversionTarget target(getContext());
    target.addLegalDialect<DFCIRDialect>();
    target.addIllegalOp<KernelOp>();
    target.addLegalDialect<circt::firrtl::FIRRTLDialect>();

    // TODO: Implement 'FIRRTLTypeConverter' completely.
    FIRRTLTypeConverter typeConverter;
    ConvertedOps convertedOps;
    OffsetMap offsetMap;
    OldTypeMap oldTypeMap;

    ModuleArgMap moduleArgMap;

    // Convert the kernel first to get a FIRRTL-circuit.
    RewritePatternSet patterns(&getContext());

    patterns.add<KernelOpConversionPattern>(
            &getContext(),
            typeConverter,
            &convertedOps,
            latencyConfig,
            &offsetMap,
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
    patterns.add<
            InputOpConversionPattern,
            ScalarInputOpConversionPattern,
            OutputOpConversionPattern,
            ScalarOutputOpConversionPattern,
            ConstantOpConversionPattern,
            OffsetOpConversionPattern,
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
            MoreOpConversionPattern,
            MoreEqOpConversionPattern,
            EqOpConversionPattern,
            NotEqOpConversionPattern,
            ConnectOpConversionPattern>(
            &getContext(),
            typeConverter,
            &convertedOps,
            latencyConfig,
            &offsetMap,
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

std::unique_ptr<mlir::Pass> createDFCIRToFIRRTLPass(LatencyConfig *config) {
  DFCIRToFIRRTLPassOptions options;
  options.latencyConfig = config;
  return std::make_unique<DFCIRToFIRRTLPass>(options);
}

} // namespace mlir::dfcir