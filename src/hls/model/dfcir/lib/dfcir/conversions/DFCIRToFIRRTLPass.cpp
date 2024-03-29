#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/conversions/DFCIRPassesUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::dfcir {

#define GEN_PASS_DECL_DFCIRTOFIRRTLPASS
#define GEN_PASS_DEF_DFCIRTOFIRRTLPASS
#include "dfcir/conversions/DFCIRPasses.h.inc"

    class FIRRTLTypeConverter : public TypeConverter {
    public:
        FIRRTLTypeConverter() {
            addConversion([](Type type) -> Type { return type; });
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

        mutable ConvertedOps *convertedOps;
        const LatencyConfig *latencyConfig;

        FIRRTLOpConversionPattern(MLIRContext *context,
                                  TypeConverter &typeConverter,
                                  ConvertedOps *convertedOps,
                                  LatencyConfig *latencyConfig)
                : OpConversionPattern<OperationType>(typeConverter, context),
                  convertedOps(convertedOps),
                  latencyConfig(latencyConfig) {
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
        LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            auto circuitOp = rewriter.create<circt::firrtl::CircuitOp>(
                    kernelOp.getLoc(),
                    rewriter.getStringAttr(kernelOp.getName()));
            auto save = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointToStart(circuitOp.getBodyBlock());
            auto fModuleOp = rewriter.create<circt::firrtl::FModuleOp>(
                    rewriter.getUnknownLoc(),
                    StringAttr::get(rewriter.getContext(), kernelOp.getName()),
                    circt::firrtl::ConventionAttr::get(rewriter.getContext(), Convention::Internal),
                    llvm::ArrayRef<circt::firrtl::PortInfo>());
            auto &kernel_region = kernelOp.getBodyRegion();
            assert(&kernel_region.getBlocks().front() != rewriter.getInsertionBlock());
            rewriter.mergeBlocks(&kernel_region.getBlocks().front(),
                                 fModuleOp.getBodyBlock(),
                                 fModuleOp.getBodyBlock()->getArguments());
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
            // Build a WireOp based on the DFCIRStreamType from InputOp.
            auto newOp = rewriter.create<circt::firrtl::WireOp>(
                    inputOp.getLoc(),
                    getTypeConverter()->convertType(inputOp->getResultTypes().front())
                    // TODO: change to pass other operands.
                    );
            rewriter.replaceOp(inputOp, newOp);
            return mlir::success();
        }
    };

    template<typename OperationType, typename AdaptorType>
    class SchedulableOpConversionPattern {
        using FExtModuleOp = circt::firrtl::FExtModuleOp;
        using InstanceOp = circt::firrtl::InstanceOp;
        using CircuitOp = circt::firrtl::CircuitOp;


        virtual std::string constructModuleName(const OperationType &op, AdaptorType &adaptor, unsigned latency) const = 0;
        virtual FExtModuleOp createModule(const std::string &name, const OperationType &op, AdaptorType &adaptor, unsigned latency, ConversionPatternRewriter &rewriter) const = 0;
        virtual void remapUses(OperationType &oldOp, AdaptorType &adaptor, InstanceOp &newOp, ConversionPatternRewriter &rewriter) const = 0;

    protected:
        FExtModuleOp findOrCreateModule(const OperationType &op, AdaptorType &adaptor,
                                        unsigned latency, ConversionPatternRewriter &rewriter) const {
            std::string moduleName = constructModuleName(op, adaptor, latency);
            CircuitOp circuit = circt::firrtl::utils::findCircuit(op);

            auto foundModule = circuit.template lookupSymbol<FExtModuleOp>(moduleName);
            if (foundModule) return foundModule;

            auto saved = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointToStart(circuit.getBodyBlock());
            FExtModuleOp newModule = createModule(moduleName, op, adaptor, latency, rewriter);
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

        std::string constructModuleName(const AddOp &op, OpAdaptor &adaptor, unsigned latency) const override {
            std::string name = ADD_MODULE"_";
            llvm::raw_string_ostream nameStream(name);
            adaptor.getFirst().getType().print(nameStream);
            nameStream << "_";
            adaptor.getSecond().getType().print(nameStream);
            nameStream << "##" << latency;
            return name;
        }

        FExtModuleOp createModule(const std::string &name, const AddOp &op, OpAdaptor &adaptor,
                                  unsigned latency, ConversionPatternRewriter &rewriter) const override {
            // TODO: Add inference.
            SmallVector<circt::firrtl::PortInfo> ports = {
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "out"),
                            getTypeConverter()->convertType(op->getResult(0).getType()),
                            circt::firrtl::Direction::Out),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "first"),
                            adaptor.getFirst().getType(),
                            circt::firrtl::Direction::In),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "second"),
                            adaptor.getSecond().getType(),
                            circt::firrtl::Direction::In)
            };
            IntegerType attrType = mlir::IntegerType::get(rewriter.getContext(), 32, mlir::IntegerType::Unsigned);
            return rewriter.create<FExtModuleOp>(
                    rewriter.getUnknownLoc(),
                    mlir::StringAttr::get(rewriter.getContext(), name),
                    circt::firrtl::ConventionAttr::get(rewriter.getContext(), Convention::Internal),
                    ports,
                    StringRef(ADD_MODULE),
                    mlir::ArrayAttr(),
                    mlir::ArrayAttr::get(
                            rewriter.getContext(),
                            circt::firrtl::ParamDeclAttr::get(rewriter.getContext(),
                                               mlir::StringAttr::get(rewriter.getContext(), BUF_MODULE_STAGES),
                                               attrType,
                                               mlir::IntegerAttr::get(attrType, latency))));
        }

        void remapUses(AddOp &oldOp, OpAdaptor &adaptor,
                       InstanceOp &newOp, ConversionPatternRewriter &rewriter) const override {
            rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), newOp.getResult(1), adaptor.getFirst());
            rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), newOp.getResult(2), adaptor.getSecond());
            for (auto &operand : llvm::make_early_inc_range(oldOp.getRes().getUses())) {
                operand.set(newOp.getResult(0));
            }
        }


        LogicalResult matchAndRewrite(AddOp addOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            unsigned latency = latencyConfig->find(ADD)->second;
            FExtModuleOp module = findOrCreateModule(addOp, adaptor, latency, rewriter);

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

        std::string constructModuleName(const MulOp &op, OpAdaptor &adaptor, unsigned latency) const override {
            std::string name = MUL_MODULE"_";
            llvm::raw_string_ostream nameStream(name);
            getTypeConverter()->convertType(op->getOperand(0).getType()).print(nameStream);
            nameStream << "_";
            getTypeConverter()->convertType(op->getOperand(1).getType()).print(nameStream);
            nameStream << "_";
            getTypeConverter()->convertType(op->getResult(0).getType()).print(nameStream);
            nameStream << "##" << latency;
            return name;
        }

        FExtModuleOp createModule(const std::string &name, const MulOp &op, OpAdaptor &adaptor,
                                  unsigned latency, ConversionPatternRewriter &rewriter) const override {
            // TODO: Add inference.
            SmallVector<circt::firrtl::PortInfo> ports = {
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "out"),
                            getTypeConverter()->convertType(op->getResult(0).getType()),
                            circt::firrtl::Direction::Out),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "first"),
                            adaptor.getFirst().getType(),
                            circt::firrtl::Direction::In),
                    circt::firrtl::PortInfo(
                            mlir::StringAttr::get(rewriter.getContext(), "second"),
                            adaptor.getSecond().getType(),
                            circt::firrtl::Direction::In)
            };
            IntegerType attrType = mlir::IntegerType::get(rewriter.getContext(), 32, mlir::IntegerType::Unsigned);
            return rewriter.create<FExtModuleOp>(
                    rewriter.getUnknownLoc(),
                    mlir::StringAttr::get(rewriter.getContext(), name),
                    circt::firrtl::ConventionAttr::get(rewriter.getContext(), Convention::Internal),
                    ports,
                    StringRef(MUL_MODULE),
                    mlir::ArrayAttr(),
                    mlir::ArrayAttr::get(
                            rewriter.getContext(),
                            circt::firrtl::ParamDeclAttr::get(rewriter.getContext(),
                                                              mlir::StringAttr::get(rewriter.getContext(), BUF_MODULE_STAGES),
                                                              attrType,
                                                              mlir::IntegerAttr::get(attrType, latency))));
        }

        void remapUses(MulOp &oldOp, OpAdaptor &adaptor, InstanceOp &newOp, ConversionPatternRewriter &rewriter) const override {
            rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), newOp.getResult(1), adaptor.getFirst());
            rewriter.create<ConnectOp>(rewriter.getUnknownLoc(), newOp.getResult(2), adaptor.getSecond());
            for (auto &operand : llvm::make_early_inc_range(oldOp.getRes().getUses())) {
                operand.set(newOp.getResult(0));
            }
        }


        LogicalResult matchAndRewrite(MulOp mulOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            unsigned latency = latencyConfig->find(MUL)->second;
            FExtModuleOp module = findOrCreateModule(mulOp, adaptor, latency, rewriter);

            InstanceOp newOp = rewriter.create<InstanceOp>(
                    mulOp.getLoc(),
                    module,
                    "placeholder");

            remapUses(mulOp, adaptor, newOp, rewriter);

            rewriter.eraseOp(mulOp);

            return mlir::success();
        }
    };

    class OutputOpConversionPattern : public FIRRTLOpConversionPattern<OutputOp> {
    public:
        using FIRRTLOpConversionPattern<OutputOp>::FIRRTLOpConversionPattern;
        using OpAdaptor = typename OutputOp::Adaptor;

        LogicalResult matchAndRewrite(OutputOp outputOp, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter) const override {
            auto newOp = rewriter.create<circt::firrtl::WireOp>(
                    outputOp.getLoc(),
                    getTypeConverter()->convertType(outputOp->getResultTypes().front())
                    // TODO: change to pass other operands.
            );

            rewriter.replaceOp(outputOp, newOp);
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

    class DFCIRToFIRRTLPass : public impl::DFCIRToFIRRTLPassBase<DFCIRToFIRRTLPass> {
    public:
        using ConvertedOps = mlir::DenseSet<mlir::Operation *>;

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

            // Convert the kernel first to get a FIRRTL-circuit.
            RewritePatternSet patterns(&getContext());

            patterns.add<KernelOpConversionPattern>(
                    &getContext(),
                    typeConverter,
                    &convertedOps,
                    latencyConfig);

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
            patterns.add<InputOpConversionPattern,
                    AddOpConversionPattern,
                    MulOpConversionPattern,
                    OutputOpConversionPattern,
                    ConnectOpConversionPattern>(
                    &getContext(),
                    typeConverter,
                    &convertedOps,
                    latencyConfig);

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