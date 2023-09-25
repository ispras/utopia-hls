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
#include "mlir/IR/Dialect.h"

#include "mlir/Transforms/DialectConversion.h"

using ArrayAttr = mlir::ArrayAttr;
template<typename Type>
using ArrayRef = mlir::ArrayRef<Type>;
using Attribute = mlir::Attribute;
using ChansOp = mlir::hil::ChansOp;
using ChanOp = mlir::hil::ChanOp;
using CircuitOp = circt::firrtl::CircuitOp;
using ClockType = circt::firrtl::ClockType;
using ConsOp = mlir::hil::ConsOp;
using ConOp = mlir::hil::ConOp;
using ConnectOp = circt::firrtl::ConnectOp;
using ConventionAttr = circt::firrtl::ConventionAttr;
using ConversionPattern = mlir::ConversionPattern;
using ConversionPatternRewriter = mlir::ConversionPatternRewriter;
using ConversionTarget = mlir::ConversionTarget;
using ConvertedOps = mlir::DenseSet<mlir::Operation *>;
using DialectRegistry = mlir::DialectRegistry;
using Direction = circt::firrtl::Direction;
using FExtModuleOp = circt::firrtl::FExtModuleOp;
using FModuleLike = circt::firrtl::FModuleLike;
using FModuleOp = circt::firrtl::FModuleOp;
using FIRRTLDialect = circt::firrtl::FIRRTLDialect;
using HILDialect = mlir::hil::HILDialect;
using InstanceOp = circt::firrtl::InstanceOp;
using IntegerType = circt::IntegerType;
using LLVMStringLiteral = llvm::StringLiteral;
using LLVMStringRef = llvm::StringRef;
using LogicalResult = mlir::LogicalResult;
using MLIRContext = mlir::MLIRContext;
using ModelOp = mlir::hil::ModelOp;
using ModuleOp = mlir::ModuleOp;
using NameKindEnum = circt::firrtl::NameKindEnum;
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
using PortInfo = circt::firrtl::PortInfo;
using Region = mlir::Region;
using ResetType = circt::firrtl::ResetType;
using RewritePattern = mlir::RewritePattern;
using RewritePatternSet = mlir::RewritePatternSet;
template<typename Type>
using SmallVector = mlir::SmallVector<Type>;
using StringAttr = mlir::StringAttr;
using TypeConverter = mlir::TypeConverter;
using TypeID = mlir::TypeID;
using Type = mlir::Type;
using Value = mlir::Value;
using WalkResult = circt::WalkResult;

namespace {

std::optional<InstanceOp> findInstance(FModuleOp fModuleOp,
                                       const std::string &name) {
  for (auto instanceOp : fModuleOp.getBodyBlock()->getOps<InstanceOp>()) {
    if (instanceOp.getName() == name) {
      return instanceOp;
    }
  }
  return std::nullopt;
}

std::optional<FExtModuleOp> findFExtModule(CircuitOp circuitOp,
                                           const std::string &name) {
  for (auto fExtModuleOp : circuitOp.getBodyBlock()->getOps<FExtModuleOp>()) {
    if (fExtModuleOp.getName() == name) {
      return fExtModuleOp;
    }
  }
  return std::nullopt;
}

std::optional<FModuleOp> findFModule(CircuitOp circuitOp,
                                     const std::string &name) {
  for (auto fModuleOp : circuitOp.getBodyBlock()->getOps<FModuleOp>()) {
    if (fModuleOp.getName() == name) {
      return fModuleOp;
    }
  }
  return std::nullopt;
}

std::optional<Value> findFModulePort(FModuleOp fModuleOp,
                                     const std::string &name) {
  auto &&portNames = fModuleOp.getPortNames();
  for (std::size_t i = 0; i < portNames.size(); i++) {
    if (mlir::cast<StringAttr>(portNames[i]).str() == name) {
      return fModuleOp.getBodyBlock()->getArgument(i);
    }
  }
  return std::nullopt;
}

std::optional<std::size_t> findFExtModulePortNumber(FExtModuleOp fExtModuleOp,
                                                    const std::string &name) {
  auto &&portNames = fExtModuleOp.getPortNames();
  for (std::size_t i = 0; i < portNames.size(); i++) {
    if (mlir::cast<StringAttr>(portNames[i]).str() == name) {
      return i;
    }
  }                                         
  return std::nullopt;
}

std::optional<std::size_t> findFModulePortNumber(FModuleOp fModuleOp,
                                                 const std::string &name) {
  auto &&portNames = fModuleOp.getPortNames();
  for (std::size_t i = 0; i < portNames.size(); i++) {
    if (mlir::cast<StringAttr>(portNames[i]).str() == name) {
      return i;
    }
  }                                         
  return std::nullopt;
}

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
    auto circuitOp = rewriter.create<CircuitOp>(
        modelOp.getLoc(), rewriter.getStringAttr(modelOp.getName()));

    auto &region = modelOp->getRegions().front();
    rewriter.mergeBlocks(&region.getBlocks().front(), circuitOp.getBodyBlock(),
        circuitOp.getBodyBlock()->getArguments());
    rewriter.eraseOp(modelOp);
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
    auto circuitOp = mlir::cast<CircuitOp>(nodeTypesOp->getParentOp());

    auto &region = nodeTypesOp->getRegions().front();
    rewriter.mergeBlocks(&region.getBlocks().front(), circuitOp.getBodyBlock(),
        circuitOp.getBodyBlock()->getArguments());
    rewriter.eraseOp(nodeTypesOp);
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
    // Sinks and sources will be removed so there is no need for the nodetypes.
    if (isSink(nodeTypeOp) || isSource(nodeTypeOp) || isInstance(nodeTypeOp)) {
      rewriter.eraseOp(nodeTypeOp);
      return mlir::success();
    }

    auto *context = nodeTypeOp.getContext();
    /// TODO: What does 'ConventionAttr' mean? What is it for?
    auto conventionAttr = ConventionAttr::get(context, Convention::Scalarized);
    SmallVector<PortInfo> portList;
    StringAttr name = rewriter.getStringAttr(nodeTypeOp.getName());
    StringAttr defName = rewriter.getStringAttr(nodeTypeOp.getName());
    ArrayAttr annotations;
    ArrayAttr parameters;
    ArrayAttr internalPaths;
    // Add 'clock' and 'reset' ports to the port list.
    portList.push_back(PortInfo{ rewriter.getStringAttr("clock"),
                                 ClockType::get(context),
                                 Direction::In });
    portList.push_back(PortInfo{ rewriter.getStringAttr("reset"),
                                 ResetType::get(context),
                                 Direction::In });
    // Add input ports to the port list.
    for (auto &port : nodeTypeOp.getCommandArguments()) {
      auto &&name = rewriter.getStringAttr(port.cast<PortAttr>().getName());
      /// TODO: Implement 'FIRRTLTypeConverter'.
      auto &&type = IntegerType::get(context, 16, IntegerType::Unsigned);
      auto &&direction = Direction::In;
      portList.push_back(PortInfo{ name, type, direction });
    }

    // Add output ports to the port list.
    for (auto &port : nodeTypeOp.getCommandResults()) {
      auto &&name = rewriter.getStringAttr(port.cast<PortAttr>().getName());
      /// TODO: Implement 'FIRRTLTypeConverter'.
      auto &&type = IntegerType::get(context, 16, IntegerType::Unsigned);
      auto &&direction = Direction::Out;
      portList.push_back(PortInfo{ name, type, direction });
    }

    // Create the 'FExtModuleOp' (FIRRTL external module).
    rewriter.create<FExtModuleOp>(nodeTypeOp.getLoc(), name, conventionAttr,
        portList, defName, annotations, parameters, internalPaths);
    // Delete the 'NodeTypeOp'.
    rewriter.eraseOp(nodeTypeOp);

    return mlir::success();
  }
};

class GraphsOpConversionPattern : public FIRRTLOpConversionPattern<GraphsOp> {
public:
  using FIRRTLOpConversionPattern<GraphsOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename GraphsOp::Adaptor;

  LogicalResult matchAndRewrite(GraphsOp graphsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto circuitOp = mlir::cast<CircuitOp>(graphsOp->getParentOp());

    auto &region = graphsOp->getRegions().front();
    rewriter.mergeBlocks(&region.getBlocks().front(), circuitOp.getBodyBlock(),
        circuitOp.getBodyBlock()->getArguments());
    rewriter.eraseOp(graphsOp);
    return mlir::success();
  }
};

class GraphOpConversionPattern : public FIRRTLOpConversionPattern<GraphOp> {
public:
  using FIRRTLOpConversionPattern<GraphOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename GraphOp::Adaptor;

  LogicalResult matchAndRewrite(GraphOp graphOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    /// TODO: What does 'ConventionAttr' mean? What is it for?
    auto *context = graphOp.getContext();
    auto conventionAttr = ConventionAttr::get(context, Convention::Scalarized);
    SmallVector<PortInfo> portList;
    // Add 'clock' and 'reset' ports to the port list.
    portList.push_back(PortInfo(rewriter.getStringAttr("clock"),
                                ClockType::get(context),
                                Direction::In));
    portList.push_back(PortInfo(rewriter.getStringAttr("reset"),
                                ResetType::get(context),
                                Direction::In));

    // Get outputs from source and inputs from sinks and remove the nodes.
    auto &graphOperations = graphOp.getBodyBlock()->getOperations();
    auto nodesOp = findElemByType<NodesOp>(graphOperations).value();
    nodesOp.walk([&](NodeOp nodeOp) {
      /// TODO: Can source have multiple outputs?
      if (isSource(nodeOp)) {
        for (auto &port : nodeOp.getCommandResults()) {
          auto &&name = nodeOp.getNameAttr();
          auto &&type = IntegerType::get(context, 16, IntegerType::Unsigned);
          auto &&direction = Direction::In;
          portList.push_back(PortInfo{ name, type, direction });
        }

        // Delete the 'NodeOp'.
        rewriter.eraseOp(nodeOp);
      }

      /// TODO: Can sink have multiple inputs?
      else if (isSink(nodeOp)) {
        for (auto &port : nodeOp.getCommandArguments()) {
          auto &&name = nodeOp.getNameAttr();
          auto &&type = IntegerType::get(context, 16, IntegerType::Unsigned);
          auto &&direction = Direction::Out;
          portList.push_back(PortInfo{ name, type, direction });
        }

        // Delete the 'NodeOp'.
        rewriter.eraseOp(nodeOp);
      }
    });

    // Create the 'FModuleOp' (FIRRTL module).
    auto &&fModuleOp = rewriter.create<FModuleOp>(graphOp.getLoc(),
        rewriter.getStringAttr(graphOp.getName()), conventionAttr, portList);

    // Print the assembly code of the created 'FModuleOp'.
    fModuleOp.dump();

    // Move the body of the 'GraphOp' to the 'FModuleOp'.
    auto &region = graphOp->getRegions().front();
    SmallVector<Value> noArguments;
    rewriter.mergeBlocks(&region.getBlocks().front(), fModuleOp.getBodyBlock(),
        noArguments);

    // Delete the 'GraphOp' operation.
    rewriter.eraseOp(graphOp);

    return mlir::success();
  }
};

class NodesOpConversionPattern : public FIRRTLOpConversionPattern<NodesOp> {
public:
  using FIRRTLOpConversionPattern<NodesOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename NodesOp::Adaptor;

  LogicalResult matchAndRewrite(NodesOp nodesOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto &&fModuleOp = mlir::cast<FModuleOp>(nodesOp->getParentOp());

    auto &region = nodesOp->getRegions().front();
    SmallVector<Value> noArguments;
    rewriter.mergeBlocks(&region.getBlocks().front(), fModuleOp.getBodyBlock(),
        noArguments);
    rewriter.eraseOp(nodesOp);
    return mlir::success();
  }
};

class NodeOpConversionPattern : public FIRRTLOpConversionPattern<NodeOp> {
public:
  using FIRRTLOpConversionPattern<NodeOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename NodeOp::Adaptor;

  LogicalResult matchAndRewrite(NodeOp nodeOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Direction> portDirections;
    SmallVector<Attribute> portNames;
    StringAttr innerSym = {};
    auto &&fModuleOp = nodeOp->getParentOfType<FModuleOp>();
    auto &&circuitOp = nodeOp->getParentOfType<CircuitOp>();
    auto &&moduleName = nodeOp.getNodeTypeName().str();
    std::optional<FModuleOp> module;
    std::optional<FExtModuleOp> extModule;
    bool isInst = isInstance(nodeOp);
    if (isInst) {
      module = findFModule(circuitOp, 
                           moduleName.substr(moduleName.find_last_of('_') + 1));      
    } else {
      extModule = findFExtModule(circuitOp, moduleName);
    }

    // Create the 'InstanceOp'.
    std::optional<InstanceOp> instanceOp;
    if (isInst) {
      instanceOp = rewriter.create<InstanceOp>(nodeOp.getLoc(), *module,
          rewriter.getStringAttr(nodeOp.getName()),
          NameKindEnum::InterestingName, ArrayRef<Attribute>(),
          ArrayRef<Attribute>(), false, innerSym);
    } else {
      instanceOp = rewriter.create<InstanceOp>(nodeOp.getLoc(), *extModule,
          rewriter.getStringAttr(nodeOp.getName()),
          NameKindEnum::InterestingName, ArrayRef<Attribute>(),
          ArrayRef<Attribute>(), false, innerSym);
    }

    // Print the assembly code of the created 'InstanceOp'.
    instanceOp->dump();

    // Move the body of the 'NodeOp' to the 'InstanceOp'.
    auto &region = nodeOp->getRegions().front();
    SmallVector<Value> noArguments;
    rewriter.mergeBlocks(&region.getBlocks().front(), fModuleOp.getBodyBlock(),
        noArguments);

    // Connect to 'clock' and 'reset ports from the 'FModuleOp'.
    auto &&sourceClockPort = findFModulePort(fModuleOp, "clock");
    auto &&targetClockPortNumber = (isInst ? findFModulePortNumber(*module,
                                                                   "clock") :
        findFExtModulePortNumber(*extModule, "clock"));
        
    rewriter.create<ConnectOp>(nodeOp.getLoc(),
                               instanceOp->getResult(*targetClockPortNumber),
                              *sourceClockPort);
    auto &&sourceResetPort = findFModulePort(fModuleOp, "reset");
    auto &&targetResetPortNumber = (isInst ? findFModulePortNumber(*module,
                                                                   "reset") :
        findFExtModulePortNumber(*extModule, "reset"));
    rewriter.create<ConnectOp>(nodeOp.getLoc(),
                               instanceOp->getResult(*targetResetPortNumber),
                              *sourceResetPort);

    // Delete the 'NodeOp' operation.
    rewriter.eraseOp(nodeOp);

    return mlir::success();
  }
};

class ChansOpConversionPattern : public FIRRTLOpConversionPattern<ChansOp> {
public:
  using FIRRTLOpConversionPattern<ChansOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ChansOp::Adaptor;

  LogicalResult matchAndRewrite(ChansOp chansOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto &&fModuleOp = mlir::cast<FModuleOp>(chansOp->getParentOp());

    auto &region = chansOp->getRegions().front();
    SmallVector<Value> noArguments;
    rewriter.mergeBlocks(&region.getBlocks().front(),
        fModuleOp.getBodyBlock(), noArguments);
    rewriter.eraseOp(chansOp);
    return mlir::success();
  }
};

class ChanOpConversionPattern : public FIRRTLOpConversionPattern<ChanOp> {
public:
  using FIRRTLOpConversionPattern<ChanOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ChanOp::Adaptor;

  LogicalResult matchAndRewrite(ChanOp chanOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Get the parent 'FModuleOp'.
    auto &&fModuleOp = chanOp->getParentOfType<FModuleOp>();

    // Get the source node(instance) name.
    auto &&sourceInstanceName = chanOp.getNodeFromAttr().getNodeName().str();

    // Get the source port.
    std::optional<Value> sourcePort;
    if (eda::utils::starts_with(sourceInstanceName, "source")) {
      sourcePort = findFModulePort(fModuleOp, sourceInstanceName);
    } else {
      auto &&sourcePortInTypeName =
          chanOp.getNodeFromAttr().getPort().cast<PortAttr>().getName();
      auto &&sourceInstance = findInstance(fModuleOp, sourceInstanceName);
      auto &&sourceInstanceModuleName = sourceInstance->getModuleName().str();
      auto &&circuitOp = chanOp->getParentOfType<CircuitOp>();
      auto &&sourceFExtModule = findFExtModule(circuitOp,
                                               sourceInstanceModuleName);
      auto &&sourcePortNumber = findFExtModulePortNumber(*sourceFExtModule,
                                                          sourcePortInTypeName);    
      sourcePort = sourceInstance->getResult(*sourcePortNumber);
    }

    // Get the target node(instance) name.
    auto &&targetInstanceName = chanOp.getNodeToAttr().getNodeName().str();
    // Get the target port.
    std::optional<Value> targetPort;
    if (eda::utils::starts_with(targetInstanceName, "sink")) {
      targetPort = findFModulePort(fModuleOp, targetInstanceName);
    } else { 
      auto &&targetPortInTypeName =
          chanOp.getNodeToAttr().getPort().cast<PortAttr>().getName();
      auto &&targetInstance = findInstance(fModuleOp, targetInstanceName);
      auto &&targetInstanceModuleName = targetInstance->getModuleName().str();
      auto &&circuitOp = chanOp->getParentOfType<CircuitOp>();
      auto &&targetFExtModule = findFExtModule(circuitOp,
                                               targetInstanceModuleName);
      auto &&targetPortNumber = findFExtModulePortNumber(*targetFExtModule,
                                                          targetPortInTypeName); 
      targetPort = targetInstance->getResult(*targetPortNumber);
    }

    // Create the 'ConnectOp'.
    auto &&connectOp = rewriter.create<ConnectOp>(chanOp.getLoc(),
                                                 *targetPort,
                                                 *sourcePort);

    // Print the assembly code of the created 'ConnectOp'.
    connectOp.dump();

    // Delete the 'ChanOp'.
    rewriter.eraseOp(chanOp);

    return mlir::success();
  }
};

class ConsOpConversionPattern : public FIRRTLOpConversionPattern<ConsOp> {
public:
  using FIRRTLOpConversionPattern<ConsOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ConsOp::Adaptor;

  LogicalResult matchAndRewrite(ConsOp consOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto &&fModuleOp = consOp->getParentOfType<FModuleOp>();

    auto &region = consOp->getRegions().front();
    SmallVector<Value> noArguments;
    rewriter.mergeBlocks(&region.getBlocks().front(), fModuleOp.getBodyBlock(),
        noArguments);
    rewriter.eraseOp(consOp);
    return mlir::success();
  }
};

class ConOpConversionPattern : public FIRRTLOpConversionPattern<ConOp> {
public:
  using FIRRTLOpConversionPattern<ConOp>::FIRRTLOpConversionPattern;
  using OpAdaptor = typename ConOp::Adaptor;

  LogicalResult matchAndRewrite(ConOp conOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Get the circuit.
    auto &&circuitOp = conOp->getParentOfType<CircuitOp>();

    // Get the body of the circuit.
    auto *body = circuitOp.getBodyBlock();

    // Get the the source module and the target module of the connection.
    std::optional<FModuleOp> sourceFModuleOp;
    std::optional<FModuleOp> targetFModuleOp;
    auto &&sourceBndGraph = conOp.getNodeFromAttr();
    auto &&sourceGraphName = sourceBndGraph.getGraphName();
    auto &&targetBndGraph = conOp.getNodeToAttr();
    auto &&targetGraphName = targetBndGraph.getGraphName();
    for (auto module : body->getOps<FModuleOp>()) {
      if (module.getName() == sourceGraphName) {
        sourceFModuleOp = module;
      } else if (module.getName() == targetGraphName) {
        targetFModuleOp = module;
      }
    }

    // Get the source and target port.
    std::optional<Value> sourcePort;
    std::optional<Value> targetPort;

    // Get the source channel.
    auto sourceGraphChansOp =
        findElemByType<ChansOp>(
            sourceFModuleOp->getBodyBlock()->getOperations()).value();
    auto &&sourceChanName = sourceBndGraph.getChanName().str();
    auto &&sourceChanOp = mlir::hil::findChan(sourceGraphChansOp,
                                              sourceChanName);

    // Get the target channel.
    auto targetGraphChansOp =
        findElemByType<ChansOp>(
            targetFModuleOp->getBodyBlock()->getOperations()).value();
    auto &&targetChanName = targetBndGraph.getChanName().str();
    auto &&targetChanOp = mlir::hil::findChan(targetGraphChansOp,
                                              targetChanName);

    // Get the direction of the connection.
    auto &&direction = conOp.getDirTypeName();

    if (direction == "IN") {
      // Get the target port.
      auto &&targetInstancePortName =
          targetChanOp->getNodeFrom().getNodeName().str();
      auto &&targetInstanceName = sourceChanOp->getNodeTo().getNodeName().str();
      auto &&targetInstance =
          findInstance(*sourceFModuleOp, targetInstanceName);
      auto &&targetInstanceModuleName = targetInstance->getModuleName().str();
      auto &&targetPortNumber = findFModulePortNumber(*targetFModuleOp,
                                                       targetInstancePortName);
      targetPort = targetInstance->getResult(*targetPortNumber);

      // Get the source port.
      auto &&sourceInstanceName =
          sourceChanOp->getNodeFrom().getNodeName().str();
      if (eda::utils::starts_with(sourceInstanceName, "source")) {
        sourcePort = findFModulePort(*sourceFModuleOp, sourceInstanceName);
      } else {
        auto &&sourcePortDeclName = sourceChanOp->getNodeFromAttr().getPort().
            cast<PortAttr>().getName();
        auto &&sourceInstance =
            findInstance(*sourceFModuleOp, sourceInstanceName);
        auto &&sourceInstanceModuleName = sourceInstance->getModuleName().str();
        auto &&sourceFExtModule = findFExtModule(circuitOp,
                                                 sourceInstanceModuleName);
        auto &&sourcePortNumber = findFExtModulePortNumber(*sourceFExtModule,
                                                            sourcePortDeclName);    
        sourcePort = sourceInstance->getResult(*sourcePortNumber);
      }

      // Delete the source channel.
      rewriter.eraseOp(*sourceChanOp);

    } else if (direction == "OUT") {
      // Get the source port.
      auto &&sourceInstancePortName =
          sourceChanOp->getNodeTo().getNodeName().str();
      auto &&sourceInstanceName =
          targetChanOp->getNodeFrom().getNodeName().str();
      auto &&sourceInstance = findInstance(*targetFModuleOp,
                                            sourceInstanceName);
      auto &&sourceInstanceModuleName = sourceInstance->getModuleName().str();
      auto &&sourcePortNumber = findFModulePortNumber(*sourceFModuleOp,
                                                       sourceInstancePortName);
      sourcePort = sourceInstance->getResult(*sourcePortNumber);

      // Get the target port.
      auto &&targetInstanceName = targetChanOp->getNodeTo().getNodeName().str();
      if (eda::utils::starts_with(targetInstanceName, "sink")) {
        targetPort = findFModulePort(*targetFModuleOp, targetInstanceName);
      } else {
        auto &&targetPortDeclName =
            targetChanOp->getNodeToAttr().getPort().cast<PortAttr>().getName();
        auto &&targetInstance = findInstance(*targetFModuleOp,
                                              targetInstanceName);
        auto &&targetInstanceModuleName = targetInstance->getModuleName().str();
        auto &&targetFExtModule = findFExtModule(circuitOp,
                                                 targetInstanceModuleName);
        auto &&targetPortNumber = findFExtModulePortNumber(*targetFExtModule,
                                                            targetPortDeclName);    
        targetPort = targetInstance->getResult(*targetPortNumber);
      }

      // Delete the target channel.
      rewriter.eraseOp(*targetChanOp);

    }

    // Create the connection between the FIRRTL modules. 
    auto &&connectOp = rewriter.create<ConnectOp>(conOp.getLoc(),
                                                 *targetPort,
                                                 *sourcePort);

    // Print the assembly code of the created 'ConnectOp'.
    connectOp.dump();

    // Delete the 'ConOp'.
    rewriter.eraseOp(conOp);

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
    /// Define the conversion target.
    ConversionTarget target(getContext());
    target.addIllegalDialect<HILDialect>();
    target.addLegalDialect<FIRRTLDialect>();

    /// Need to make ChanOp legal because it has to be processed after NodeOp.
    target.addLegalOp<ChansOp>();
    target.addLegalOp<ChanOp>();

    /// TODO: Implement 'FIRRTLTypeConverter'.
    FIRRTLTypeConverter typeConverter;
    ConvertedOps convertedOps;

    /// Define the first set of rewrite patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<ModelOpConversionPattern,
                 NodeTypesOpConversionPattern, NodeTypeOpConversionPattern,
                 GraphsOpConversionPattern, GraphOpConversionPattern,
                 NodesOpConversionPattern, NodeOpConversionPattern,
                 ConsOpConversionPattern, ConOpConversionPattern>(
        &getContext(), typeConverter, &convertedOps);

    /// Apply partial conversion.
    if (failed(applyPartialConversion(getOperation(),
                                      target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }

    /// Define the second set of rewrite patterns.
    patterns.clear();
    target.addIllegalOp<ChansOp>();
    target.addIllegalOp<ChanOp>();
    patterns.add<ChansOpConversionPattern, ChanOpConversionPattern>(
        &getContext(), typeConverter, &convertedOps);

    /// Apply partial conversion.
    if (failed(applyPartialConversion(getOperation(),
                                      target,
                                      std::move(patterns)))) {
      signalPassFailure();

    }
  }
};
} // namespace

std::unique_ptr<Pass> createHILToFIRRTLPass() {
  return std::make_unique<HILToFIRRTLPass>();
}