#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

template<>
struct std::hash<std::pair<mlir::Operation *, unsigned>> {
    size_t operator() (const std::pair<mlir::Operation *, unsigned> &pair) const noexcept {
        return std::hash<mlir::Operation *>()(pair.first) + std::hash<unsigned>()(pair.second);
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
            addConversion([](DFCIRScalarType type) -> circt::firrtl::IntType {
                Type scalarType = type.getScalarType();
                if (scalarType.isa<DFCIRFixedType>()) {
                    DFCIRFixedType fixedType = llvm::cast<DFCIRFixedType>(scalarType);
                    unsigned width = fixedType.getIntegerBits() + fixedType.getFractionBits();
                    if (fixedType.getSign()) {
                        return circt::firrtl::SIntType::get(fixedType.getContext(), width);
                    } else {
                        return circt::firrtl::UIntType::get(fixedType.getContext(), width);
                    }
                } else if (scalarType.isa<DFCIRFloatType>()) {
                    DFCIRFloatType floatType = llvm::cast<DFCIRFloatType>(scalarType);
                    unsigned width = floatType.getExponentBits() + floatType.getFractionBits();
                    return circt::firrtl::UIntType::get(floatType.getContext(), width);
                }
                return {};
            });
            addConversion([](DFCIRStreamType type) -> circt::firrtl::IntType {
                Type streamType = type.getStreamType();
                if (streamType.isa<DFCIRFixedType>()) {
                    DFCIRFixedType fixedType = llvm::cast<DFCIRFixedType>(streamType);
                    unsigned width = fixedType.getIntegerBits() + fixedType.getFractionBits();
                    if (fixedType.getSign()) {
                        return circt::firrtl::SIntType::get(fixedType.getContext(), width);
                    } else {
                        return circt::firrtl::UIntType::get(fixedType.getContext(), width);
                    }
                } else if (streamType.isa<DFCIRFloatType>()) {
                    DFCIRFloatType floatType = llvm::cast<DFCIRFloatType>(streamType);
                    unsigned width = floatType.getExponentBits() + floatType.getFractionBits();
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
        using OffsetMap = std::unordered_map<std::pair<mlir::Operation *, unsigned>, signed>;

        mutable ConvertedOps *convertedOps;
        const LatencyConfig *latencyConfig;
        OffsetMap *offsetMap;
        ModuleArgMap *moduleArgMap;

        FIRRTLOpConversionPattern(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  ConvertedOps *convertedOps,
                                  LatencyConfig *latencyConfig,
                                  OffsetMap *offsetMap,
                                  ModuleArgMap *moduleArgMap)
                : OpConversionPattern<OperationType>(typeConverter, context),
                  convertedOps(convertedOps),
                  latencyConfig(latencyConfig),
                  offsetMap(offsetMap),
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
        using PortInfo = circt::firrtl::PortInfo;
        using SpecialConstantOp = circt::firrtl::SpecialConstantOp;
        using ClockType = circt::firrtl::ClockType;
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
            for (Operation &op : kernelBlock->getOperations()) {
                if (auto named = llvm::dyn_cast<NamedOpVal>(op)) {
                    (*moduleArgMap)[&op] = argInd++;
                    llvm::StringRef name = named.getValueName();
                    ports.push_back(&op);
                    Type converted = getTypeConverter()->convertType(op.getResult(0).getType());
                    modulePorts.emplace_back(
                            mlir::StringAttr::get(getContext(), name),
                            converted,
                            (llvm::isa<InputOp, ScalarInputOp>(op)) ? circt::firrtl::Direction::In :
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
                    ConventionAttr::get(rewriter.getContext(), Convention::Internal),
                    modulePorts);

            // Replace the input-/output-operations' results with block arguments.

            for (size_t index = 0; index < ports.size(); ++index) {
                BlockArgument arg = fModuleOp.getArgument(index);
                for (auto &operand : llvm::make_early_inc_range(ports[index]->getResult(0).getUses())) {
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

    class ScalarInputOpConversionPattern : public FIRRTLOpConversionPattern<ScalarInputOp> {
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
                createConnect(rewriter, getBlockArgumentFromOpBlock(outputOp, (*moduleArgMap)[outputOp]), adaptor.getStream());
            }
            rewriter.eraseOp(outputOp);
            return mlir::success();
        }
    };

    class ScalarOutputOpConversionPattern : public FIRRTLOpConversionPattern<ScalarOutputOp> {
    public:
        using FIRRTLOpConversionPattern<ScalarOutputOp>::FIRRTLOpConversionPattern;
        using OpAdaptor = typename ScalarOutputOp::Adaptor;

        LogicalResult matchAndRewrite(ScalarOutputOp scalarOutputOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            using circt::firrtl::utils::createConnect;
            using circt::firrtl::utils::getBlockArgumentFromOpBlock;

            // TODO: Add control stream functionality.
            if (scalarOutputOp.getStream()) {
                createConnect(rewriter, getBlockArgumentFromOpBlock(scalarOutputOp, (*moduleArgMap)[scalarOutputOp]), adaptor.getStream());
            }
            rewriter.eraseOp(scalarOutputOp);
            return mlir::success();
        }
    };

    class ConstOpConversionPattern : public FIRRTLOpConversionPattern<ConstOp> {
    public:
        using FIRRTLOpConversionPattern<ConstOp>::FIRRTLOpConversionPattern;
        using OpAdaptor = typename ConstOp::Adaptor;

        LogicalResult matchAndRewrite(ConstOp constOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            using circt::firrtl::ConstantOp;
            using circt::firrtl::UIntType;
            using circt::firrtl::SIntType;
            using circt::firrtl::IntType;
            auto castedInt = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
            auto castedFloat = llvm::dyn_cast<mlir::FloatAttr>(constOp.getValue());
            if (castedInt) {
                int32_t width = castedInt.getType().getIntOrFloatBitWidth();
                Type newType;
                if (castedInt.getType().isSignedInteger()) {
                    newType = SIntType::get(getContext(), width, true);
                } else {
                    newType = UIntType::get(getContext(), width, true);
                }
                auto newOp = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(), newType, castedInt);
                for (auto &operand : llvm::make_early_inc_range(constOp->getResult(0).getUses())) {
                    operand.set(newOp.getResult());
                }
            } else if (castedFloat) {
                // TODO: Add float functionality.
                assert(false && "No floats yet");
            }
            rewriter.eraseOp(constOp);
            return mlir::success();
        }
    };


    template<typename OperationType, typename AdaptorType>
    class SchedulableOpConversionPattern {
        using FExtModuleOp = circt::firrtl::FExtModuleOp;
        using InstanceOp = circt::firrtl::InstanceOp;
        using CircuitOp = circt::firrtl::CircuitOp;


        virtual std::string constructModuleName(const OperationType &op, AdaptorType &adaptor) const = 0;
        virtual FExtModuleOp createModule(const std::string &name, const OperationType &op, AdaptorType &adaptor, ConversionPatternRewriter &rewriter) const = 0;
        virtual void remapUses(OperationType &oldOp, AdaptorType &adaptor, InstanceOp &newOp, ConversionPatternRewriter &rewriter) const = 0;

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

    };

    class AddOpConversionPattern : public FIRRTLOpConversionPattern<AddOp>,
                                          SchedulableOpConversionPattern<AddOp,
                                                                         AddOp::Adaptor> {
    public:
        using FIRRTLOpConversionPattern<AddOp>::FIRRTLOpConversionPattern;
        using OpAdaptor = typename AddOp::Adaptor;
        using FExtModuleOp = circt::firrtl::FExtModuleOp;
        using InstanceOp = circt::firrtl::InstanceOp;
        using ConnectOp = circt::firrtl::ConnectOp;
        using IntType = circt::firrtl::IntType;
        using SIntType = circt::firrtl::SIntType;
        using UIntType = circt::firrtl::UIntType;

        std::string constructModuleName(const AddOp &op, OpAdaptor &adaptor) const override {
            Type type = op->getResult(0).getType();
            Type convType = getTypeConverter()->convertType(type);

            bool isFloat = false;

            std::string name = ADD_MODULE"_";
            llvm::raw_string_ostream nameStream(name);

            if ((isFloat = type.isa<DFCIRFloatType>())) {
                nameStream << FLOAT_SPEC"_";
            } else if (convType.isa<IntType>()) {
                nameStream << INT_SPEC"_";
            }
            unsigned latency;
            if (isFloat) {
                DFCIRFloatType casted = llvm::cast<DFCIRFloatType>(type);
                nameStream << (casted.getExponentBits() + casted.getFractionBits()) << "#" << casted.getExponentBits();
                latency = latencyConfig->find(ADD_FLOAT)->second;
            } else {
                nameStream << llvm::cast<IntType>(convType).getWidthOrSentinel() << "_";
                latency = latencyConfig->find(ADD_INT)->second;
            }
            nameStream << "##" << latency;
            return name;
        }

        FExtModuleOp createModule(const std::string &name, const AddOp &op, OpAdaptor &adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
            Type type = op->getResult(0).getType();
            Type converted = getTypeConverter()->convertType(type);
            SmallVector<circt::firrtl::PortInfo> ports = {
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "out"),
                            converted,
                            circt::firrtl::Direction::Out),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "first"),
                            adaptor.getFirst().getType(),
                            circt::firrtl::Direction::In),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "second"),
                            adaptor.getSecond().getType(),
                            circt::firrtl::Direction::In),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "clk"),
                            circt::firrtl::ClockType::get(rewriter.getContext()),
                            circt::firrtl::Direction::In)
            };
            IntegerType attrType = mlir::IntegerType::get(rewriter.getContext(), 32, mlir::IntegerType::Unsigned);
            auto outTypeWidth = circt::firrtl::getBitWidth(llvm::dyn_cast<circt::firrtl::FIRRTLBaseType>(converted));
            assert(outTypeWidth.has_value());
            auto firstTypeWidth = circt::firrtl::getBitWidth(llvm::dyn_cast<circt::firrtl::FIRRTLBaseType>(adaptor.getFirst().getType()));
            assert(firstTypeWidth.has_value());
            auto secondTypeWidth = circt::firrtl::getBitWidth(llvm::dyn_cast<circt::firrtl::FIRRTLBaseType>(adaptor.getSecond().getType()));
            assert(secondTypeWidth.has_value());
            assert(*outTypeWidth == *firstTypeWidth && *outTypeWidth == *secondTypeWidth);

            bool isFloat = type.isa<DFCIRFloatType>();
            unsigned latency = latencyConfig->find((isFloat) ? ADD_FLOAT : ADD_INT)->second;
            auto module = rewriter.create<FExtModuleOp>(
                    rewriter.getUnknownLoc(),
                    mlir::StringAttr::get(rewriter.getContext(), name),
                    circt::firrtl::ConventionAttr::get(rewriter.getContext(), Convention::Internal),
                    ports,
                    StringRef((isFloat ? (ADD_MODULE "_" FLOAT_SPEC) : (ADD_MODULE "_" INT_SPEC))),
                    mlir::ArrayAttr(),
                    mlir::ArrayAttr::get(rewriter.getContext(),
                        {
                            circt::firrtl::ParamDeclAttr::get(rewriter.getContext(),
                                mlir::StringAttr::get(rewriter.getContext(), STAGES_PARAM),
                                attrType,
                                mlir::IntegerAttr::get(attrType, latency)),
                            circt::firrtl::ParamDeclAttr::get(rewriter.getContext(),
                                mlir::StringAttr::get(rewriter.getContext(), "op_" TYPE_SIZE_PARAM),
                                attrType,
                                mlir::IntegerAttr::get(attrType, *outTypeWidth))
                        }));
            module->setAttr(INSTANCE_LATENCY_ATTR, mlir::IntegerAttr::get(attrType, latency));
            return module;
        }

        void remapUses(AddOp &oldOp, OpAdaptor &adaptor,
                       InstanceOp &newOp, ConversionPatternRewriter &rewriter) const override {
            using circt::firrtl::utils::createConnect;
            using circt::firrtl::utils::getClockVarFromOpBlock;
            createConnect(rewriter, newOp.getResult(1), adaptor.getFirst(), (*offsetMap)[std::make_pair(oldOp, 0)]);
            createConnect(rewriter, newOp.getResult(2), adaptor.getSecond(), (*offsetMap)[std::make_pair(oldOp, 1)]);
            createConnect(rewriter, newOp.getResult(3), getClockVarFromOpBlock(newOp));

            for (auto &operand : llvm::make_early_inc_range(oldOp.getRes().getUses())) {
                operand.set(newOp.getResult(0));
            }
        }


        LogicalResult matchAndRewrite(AddOp addOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            FExtModuleOp module = findOrCreateModule(addOp, adaptor, rewriter);

            InstanceOp newOp = rewriter.create<InstanceOp>(
                    addOp.getLoc(),
                    module,
                    "placeholder");

            remapUses(addOp, adaptor, newOp, rewriter);

            rewriter.eraseOp(addOp);

            return mlir::success();
        }
    };

    class MulOpConversionPattern : public FIRRTLOpConversionPattern<MulOp>,
                                          SchedulableOpConversionPattern<MulOp, MulOp::Adaptor> {
    public:
        using FIRRTLOpConversionPattern<MulOp>::FIRRTLOpConversionPattern;
        using OpAdaptor = typename MulOp::Adaptor;
        using FExtModuleOp = circt::firrtl::FExtModuleOp;
        using InstanceOp = circt::firrtl::InstanceOp;
        using ConnectOp = circt::firrtl::ConnectOp;
        using IntType = circt::firrtl::IntType;

        std::string constructModuleName(const MulOp &op, OpAdaptor &adaptor) const override {
            Type type = op->getResult(0).getType();
            Type convType = getTypeConverter()->convertType(type);

            bool isFloat = false;

            std::string name = MUL_MODULE"_";
            llvm::raw_string_ostream nameStream(name);

            if ((isFloat = type.isa<DFCIRFloatType>())) {
                nameStream << FLOAT_SPEC"_";
            } else if (convType.isa<IntType>()) {
                nameStream << INT_SPEC"_";
            }

            unsigned latency;
            if (isFloat) {
                DFCIRFloatType casted = llvm::cast<DFCIRFloatType>(type);
                nameStream << (casted.getExponentBits() + casted.getFractionBits()) << "#" << casted.getExponentBits();
                latency = latencyConfig->find(MUL_FLOAT)->second;
            } else {
                nameStream << llvm::cast<IntType>(convType).getWidthOrSentinel() << "_";
                latency = latencyConfig->find(MUL_INT)->second;
            }

            nameStream << "##" << latency;
            return name;
        }

        FExtModuleOp createModule(const std::string &name, const MulOp &op, OpAdaptor &adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
            Type type = op->getResult(0).getType();
            Type converted = getTypeConverter()->convertType(type);
            SmallVector<circt::firrtl::PortInfo> ports = {
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "out"),
                            converted,
                            circt::firrtl::Direction::Out),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "first"),
                            adaptor.getFirst().getType(),
                            circt::firrtl::Direction::In),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "second"),
                            adaptor.getSecond().getType(),
                            circt::firrtl::Direction::In),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "clk"),
                            circt::firrtl::ClockType::get(rewriter.getContext()),
                            circt::firrtl::Direction::In)
            };
            IntegerType attrType = mlir::IntegerType::get(rewriter.getContext(), 32, mlir::IntegerType::Unsigned);
            auto outTypeWidth = circt::firrtl::getBitWidth(llvm::dyn_cast<circt::firrtl::FIRRTLBaseType>(converted));
            assert(outTypeWidth.has_value());
            auto firstTypeWidth = circt::firrtl::getBitWidth(llvm::dyn_cast<circt::firrtl::FIRRTLBaseType>(adaptor.getFirst().getType()));
            assert(firstTypeWidth.has_value());
            auto secondTypeWidth = circt::firrtl::getBitWidth(llvm::dyn_cast<circt::firrtl::FIRRTLBaseType>(adaptor.getSecond().getType()));
            assert(secondTypeWidth.has_value());
            assert(*outTypeWidth == *firstTypeWidth && *outTypeWidth == *secondTypeWidth);

            bool isFloat = type.isa<DFCIRFloatType>();
            unsigned latency = latencyConfig->find((isFloat) ? MUL_FLOAT : MUL_INT)->second;
            auto module = rewriter.create<FExtModuleOp>(
                    rewriter.getUnknownLoc(),
                    mlir::StringAttr::get(rewriter.getContext(), name),
                    circt::firrtl::ConventionAttr::get(rewriter.getContext(), Convention::Internal),
                    ports,
                    StringRef((isFloat ? (MUL_MODULE "_" FLOAT_SPEC) : (MUL_MODULE "_" INT_SPEC))),
                    mlir::ArrayAttr(),
                    mlir::ArrayAttr::get(rewriter.getContext(),
                                         {
                                                 circt::firrtl::ParamDeclAttr::get(rewriter.getContext(),
                                                                                   mlir::StringAttr::get(rewriter.getContext(), STAGES_PARAM),
                                                                                   attrType,
                                                                                   mlir::IntegerAttr::get(attrType, latency)),
                                                 circt::firrtl::ParamDeclAttr::get(rewriter.getContext(),
                                                                                   mlir::StringAttr::get(rewriter.getContext(), "op_" TYPE_SIZE_PARAM),
                                                                                   attrType,
                                                                                   mlir::IntegerAttr::get(attrType, *outTypeWidth))
                                         }));
            module->setAttr(INSTANCE_LATENCY_ATTR, mlir::IntegerAttr::get(attrType, latency));
            return module;
        }

        void remapUses(MulOp &oldOp, OpAdaptor &adaptor, InstanceOp &newOp, ConversionPatternRewriter &rewriter) const override {
            using circt::firrtl::utils::createConnect;
            using circt::firrtl::utils::getClockVarFromOpBlock;
            createConnect(rewriter, newOp.getResult(1), adaptor.getFirst(), (*offsetMap)[std::make_pair(oldOp, 0)]);
            createConnect(rewriter, newOp.getResult(2), adaptor.getSecond(), (*offsetMap)[std::make_pair(oldOp, 1)]);
            createConnect(rewriter, newOp.getResult(3), getClockVarFromOpBlock(newOp));

            for (auto &operand : llvm::make_early_inc_range(oldOp.getRes().getUses())) {
                operand.set(newOp.getResult(0));
            }
        }


        LogicalResult matchAndRewrite(MulOp mulOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            // unsigned latency = latencyConfig->find(MUL)->second;
            FExtModuleOp module = findOrCreateModule(mulOp, adaptor, rewriter);

            InstanceOp newOp = rewriter.create<InstanceOp>(
                    mulOp.getLoc(),
                    module,
                    "placeholder");

            remapUses(mulOp, adaptor, newOp, rewriter);

            rewriter.eraseOp(mulOp);

            return mlir::success();
        }
    };

    class ConnectOpConversionPattern : public FIRRTLOpConversionPattern<ConnectOp> {
    public:
        using FIRRTLOpConversionPattern<ConnectOp>::FIRRTLOpConversionPattern;
        using OpAdaptor = typename ConnectOp::Adaptor;

        LogicalResult matchAndRewrite(ConnectOp connectOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            auto newOp = rewriter.create<circt::firrtl::ConnectOp>(
                    connectOp.getLoc(),
                    adaptor.getConnecting(),
                    adaptor.getConnectee()
            );
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

            for (auto &operand : llvm::make_early_inc_range(offsetOp.getRes().getUses())) {
                operand.set(offsetOp.getOperand());
                (*offsetMap)[std::make_pair(operand.getOwner(), operand.getOperandNumber())] = offset;
            }

            rewriter.eraseOp(offsetOp);
            return mlir::success();
        }
    };

    class DFCIRToFIRRTLPass : public impl::DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass> {
    public:
        using ConvertedOps = mlir::DenseSet<mlir::Operation *>;
        using OffsetMap = std::unordered_map<std::pair<mlir::Operation *, unsigned>, signed>;

        explicit DFCIRToFIRRTLPass(const DFCIRToFIRRTLPassOptions &options)
                : impl::DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass>(options) { }

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
            ModuleArgMap moduleArgMap;

            // Convert the kernel first to get a FIRRTL-circuit.
            RewritePatternSet patterns(&getContext());

            patterns.add<KernelOpConversionPattern>(
                    &getContext(),
                    typeConverter,
                    &convertedOps,
                    latencyConfig,
                    &offsetMap,
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
                    ConstOpConversionPattern,
                    OffsetOpConversionPattern,
                    AddOpConversionPattern,
                    MulOpConversionPattern,
                    ConnectOpConversionPattern>(
                    &getContext(),
                    typeConverter,
                    &convertedOps,
                    latencyConfig,
                    &offsetMap,
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