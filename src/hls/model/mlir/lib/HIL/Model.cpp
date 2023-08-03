//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
// MLIR model.
//
//===----------------------------------------------------------------------===//

#include "HIL/Model.h"

#include "HIL/Dialect.h"
#include "HIL/Dumper.h"
#include "HIL/Ops.h"
#include "HIL/Utils.h"
#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"
#include "mlir/InitAllDialects.h"

#include <fstream>
#include <iostream>
#include <optional>

using Builder = eda::hls::parser::hil::Builder;

namespace mlir::model {
MLIRModule MLIRModule::loadFromMlir(const std::string &string) {
  auto context = std::make_unique<MLIRContext>();
  context->getOrLoadDialect<HILDialect>();
  auto module = mlir::parseSourceString<ModuleOp>(string, context.get());
  return {std::move(context), std::move(module)};
}

MLIRModule MLIRModule::loadFromMlirFile(const std::string &filename) {
  std::ifstream file{filename};
  std::stringstream buf;
  if (file) {
    buf << file.rdbuf();
  } else { // fallback to stdin
    buf << std::cin.rdbuf();
  }
  return loadFromMlir(buf.str());
}

MLIRModule MLIRModule::loadFromModel(const Model &model) {
  std::stringstream buf;
  dumpModelMlir(model, buf);

  return loadFromMlir(buf.str());
}

void MLIRModule::print(llvm::raw_ostream &os) { module->print(os); }

ModelOp MLIRModule::getRoot() {
  return mlir::cast<ModelOp>(*module->getOps().begin());
}

MLIRModule::MLIRModule(MLIRModule &&oth)
    : context(std::move(oth.context)), module(std::move(oth.module)) {}

MLIRModule &MLIRModule::operator=(MLIRModule &&oth) {
  module = std::move(oth.module);
  context = std::move(oth.context);
  return *this;
}

MLIRModule MLIRModule::clone() { return {context, module->clone()}; }

MLIRContext *MLIRModule::getContext() { return module->getContext(); }

MLIRModule::MLIRModule(std::shared_ptr<MLIRContext> context,
                       mlir::OwningOpRef<ModuleOp> &&module)
    : context(std::move(context)), module(std::move(module)) {}
} // namespace mlir::model

namespace {
using mlir::model::MLIRModule;

template <typename T> class MLIRBuilder {
public:
  MLIRBuilder(T &node, Builder &builder) : node(node), builder(builder) {}
  MLIRBuilder(T &&node, Builder &builder) : node(node), builder(builder) {}
  void build();
  template <typename U>
  static MLIRBuilder<std::decay_t<U>> get(U &&node, Builder &builder) {
    return MLIRBuilder<std::decay_t<U>>(std::forward<U>(node), builder);
  }
  static std::shared_ptr<Model> buildModelFromMlir(MLIRModule &mlirModule,
                                                   Builder &builder);

private:
  T &node;
  Builder &builder;
};

template <typename T>
std::shared_ptr<Model>
MLIRBuilder<T>::buildModelFromMlir(MLIRModule &mlirModule,
                                   Builder &builder) {
  MLIRBuilder(mlirModule.getRoot(), builder).build();
  return builder.create();
}

template <> void MLIRBuilder<PortAttr>::build() {
  auto isConst = node.getIsConst();
  if (isConst == 0) {
    builder.addPort(node.getName(), node.getTypeName(),
                    std::to_string(node.getFlow()),
                    std::to_string(node.getLatency()));
  } else {
    auto value = node.getValue();
    builder.addPort(node.getName(), node.getTypeName(),
                    std::to_string(node.getFlow()),
                    std::to_string(node.getLatency()),
                    std::to_string(value));
  }
}

template <> void MLIRBuilder<NodeTypeOp>::build() {
  builder.startNodetype(node.getName().str());
  // Build inputs
  for (auto op : node.getCommandArguments()) {
    auto inPortAttr = op.cast<mlir::Attribute>().cast<PortAttr>();
    MLIRBuilder::get(inPortAttr, builder).build();
  }
  // Build outputs
  builder.startOutputs();
  for (auto op : node.getCommandResults()) {
    auto outPortAttr = op.cast<mlir::Attribute>().cast<PortAttr>();
    MLIRBuilder::get(outPortAttr, builder).build();
  }
  builder.endNodetype();
}

template <> void MLIRBuilder<NodeTypesOp>::build() {
  for (auto &op : node.getBodyBlock()->getOperations()) {
    auto nodeTypeOp = mlir::cast<NodeTypeOp>(op);
    MLIRBuilder::get(nodeTypeOp, builder).build();
  }
}

template <> void MLIRBuilder<ChanOp>::build() {
  auto nodeFrom = node.getNodeFrom();
  auto nodeTo = node.getNodeTo();
  auto typeName = node.getTypeName().str();
  auto varName = node.getVarName().str();
  builder.addChan(typeName, varName, nodeFrom, nodeTo);
}

template <> void MLIRBuilder<ChansOp>::build() {
  for (auto &op : node.getBodyBlock()->getOperations()) {
    auto chanOp = mlir::cast<ChanOp>(op);
    MLIRBuilder::get(chanOp, builder).build();
  }
}

template <> void MLIRBuilder<ConOp>::build() {
  auto nodeFrom = node.getNodeFrom();
  auto nodeTo = node.getNodeTo();
  auto varName = node.getVarName().str();
  auto typeName = node.getTypeName().str();
  auto dirTypeName = node.getDirTypeName().str();
  builder.addCon(varName, typeName, dirTypeName, nodeFrom, nodeTo);
}

template <> void MLIRBuilder<ConsOp>::build() {
  for (auto &op : node.getBodyBlock()->getOperations()) {
    auto conOp = mlir::cast<ConOp>(op);
    MLIRBuilder::get(conOp, builder).build();
  }
}

template <> void MLIRBuilder<NodeOp>::build() {
  // Get Signature of the Nodetype to look in Nodetype storage.
  const auto nodeTypeName = node.getNodeTypeName().str();
  std::vector<std::string> inputChanNames;
  for (auto op : node.getCommandArguments()) {
    inputChanNames.push_back(op.cast<mlir::StringAttr>().getValue().str());
  }
  std::vector<std::string> outputChanNames;
  for (auto op : node.getCommandResults()) {
    outputChanNames.push_back(op.cast<mlir::StringAttr>().getValue().str());
  }
  Signature signature = builder.getNodeTypeSignature(nodeTypeName,
                                                     inputChanNames,
                                                     outputChanNames);

  builder.startNode(signature, node.getName().str());

  for (auto op : node.getCommandArguments()) {
    auto chanName = op.cast<mlir::StringAttr>().getValue().str();
    builder.addParam(chanName);
  }
  builder.startOutputs();
  for (auto op : node.getCommandResults()) {
    auto chanName = op.cast<mlir::StringAttr>().getValue().str();
    builder.addParam(chanName);
  }

  builder.endNode();

  if (isInstance(node)) {
    builder.startInst(node.getName().str());
    auto &operations = node.getBodyBlock()->getOperations();
    auto consOp = findElemByType<ConsOp>(operations.begin(), operations.end());
    if (consOp) {
      MLIRBuilder::get(*consOp, builder).build();
    } else {
      std::cerr << "ERROR: `Cons` operation not found!\n";
      exit(1);
    }
    builder.endInst();
  }
}

template <> void MLIRBuilder<NodesOp>::build() {
  for (auto &op : node.getBodyBlock()->getOperations()) {
    auto nodeOp = mlir::cast<NodeOp>(op);
    MLIRBuilder::get(nodeOp, builder).build();
  }
}

template <> void MLIRBuilder<GraphOp>::build() {
  builder.startGraph(node.getName().str());
  auto &operations = node.getBodyBlock()->getOperations();
  auto chansOp = findElemByType<ChansOp>(operations.begin(), operations.end());
  if (chansOp) {
    MLIRBuilder::get(*chansOp, builder).build();
  } else {
    std::cerr << "ERROR: `Chans` operation not found!\n";
    exit(1);
  }
  auto nodesOp = findElemByType<NodesOp>(operations.begin(),
                                         operations.end());
  if (nodesOp) {
    MLIRBuilder::get(*nodesOp, builder).build();
  } else {
    std::cerr << "ERROR: `Nodes` operation not found!\n";
    exit(1);
  }
  builder.endGraph();
}

template <> void MLIRBuilder<GraphsOp>::build() {
  for (auto &operation : node.getBodyBlock()->getOperations()) {
    auto graphOp = mlir::cast<GraphOp>(operation);
    MLIRBuilder::get(graphOp, builder).build();
  }
}

template <> void MLIRBuilder<ModelOp>::build() {
  builder.startModel(node.getName().str());
  auto &operations = node.getBodyBlock()->getOperations();
  auto nodeTypesOp = findElemByType<NodeTypesOp>(operations.begin(),
                                                 operations.end());
  if (nodeTypesOp) {
    MLIRBuilder::get(*nodeTypesOp, builder).build();
  } else {
    std::cerr << "ERROR: `NodeTypes` operation not found!\n";
    exit(1);
  }
  auto graphsOp = findElemByType<GraphsOp>(operations.begin(),
                                           operations.end());
  if (graphsOp) {
    MLIRBuilder::get(*graphsOp, builder).build();
  } else {
    std::cerr << "ERROR: `Graphs` operation not found!\n";
    exit(1);
  }
  builder.endModel();
}
} // namespace

namespace eda::hls::model {

std::shared_ptr<Model> parseModelFromMlir(const std::string &string) {
  auto &builder = Builder::get();
  auto mlirModelLayer = MLIRModule::loadFromMlir(string);
  auto model = MLIRBuilder<ModelOp>::buildModelFromMlir(mlirModelLayer,
                                                        builder);
  /* std::cerr << "***********MODEL_BEGIN**********" << std::endl; */
  /* std::cerr << *model << std::endl; */
  /* std::cerr << "************MODEL_END***********" << std::endl; */
  /* std::cerr << "***********MLIR_BEGIN***********" << std::endl; */
  /* dumpModelMlir(*model, std::cerr); */
  /* std::cerr << "************MLIR_END************" << std::endl; */
  return std::shared_ptr<Model>(std::move(model));
}

std::shared_ptr<Model> parseModelFromMlirFile(const std::string &filename) {
  std::ifstream file{filename};
  std::stringstream buf;
  if (file) {
    buf << file.rdbuf();
  } else { // fallback to stdin
    buf << std::cin.rdbuf();
  }
  return parseModelFromMlir(buf.str());
}

} // namespace eda::hls::model