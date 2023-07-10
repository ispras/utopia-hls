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

using eda::hls::model::Model;
using eda::hls::parser::hil::Builder;

namespace mlir::model {
MLIRModule MLIRModule::loadFromMlir(const std::string &string) {
  auto context = std::make_unique<mlir::MLIRContext>();
  context->getOrLoadDialect<mlir::hil::HILDialect>();
  context->getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(string, context.get());
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

MLIRModule MLIRModule::loadFromModel(const eda::hls::model::Model &model) {
  std::stringstream buf;
  dumpModelMlir(model, buf);

  return loadFromMlir(buf.str());
}

void MLIRModule::print(llvm::raw_ostream &os) { module->print(os); }

mlir::hil::Model MLIRModule::getRoot() {
  return mlir::cast<mlir::hil::Model>(*module->getOps().begin());
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

MLIRModule::MLIRModule(std::shared_ptr<mlir::MLIRContext> context,
                       mlir::OwningOpRef<mlir::ModuleOp> &&module)
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

template <> void MLIRBuilder<mlir::hil::PortAttr>::build() {
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

template <> void MLIRBuilder<mlir::hil::NodeType>::build() {
  builder.startNodetype(node.name().str());
  // Build inputs
  for (auto op : node.commandArguments()) {
    auto inPortOp = op.cast<mlir::Attribute>().cast<mlir::hil::PortAttr>();
    MLIRBuilder::get(inPortOp, builder).build();
  }
  // Build outputs
  builder.startOutputs();
  for (auto op : node.commandResults()) {
    auto outPortOp = op.cast<mlir::Attribute>().cast<mlir::hil::PortAttr>();
    MLIRBuilder::get(outPortOp, builder).build();
  }
  builder.endNodetype();
}

template <> void MLIRBuilder<mlir::hil::NodeTypes>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto nodetypeOp = mlir::cast<mlir::hil::NodeType>(op);
    MLIRBuilder::get(nodetypeOp, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Chan>::build() {
  auto nodeFrom = node.nodeFrom();
  auto nodeTo = node.nodeTo();
  auto typeName = node.typeName().str();
  auto varName = node.varName().str();
  builder.addChan(typeName, varName, nodeFrom, nodeTo);
}

template <> void MLIRBuilder<mlir::hil::Chans>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto chanOp = mlir::cast<mlir::hil::Chan>(op);
    MLIRBuilder::get(chanOp, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Con>::build() {
  auto nodeFrom = node.nodeFrom();
  auto nodeTo = node.nodeTo();
  auto varName = node.varName().str();
  auto typeName = node.typeName().str();
  auto dirTypeName = node.dirTypeName().str();
  builder.addCon(varName, typeName, dirTypeName, nodeFrom, nodeTo);
}

template <> void MLIRBuilder<mlir::hil::Cons>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto conOp = mlir::cast<mlir::hil::Con>(op);
    MLIRBuilder::get(conOp, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Node>::build() {
  // Get Signature of the Nodetype to look in Nodetype storage.
  const auto nodeTypeName = node.nodeTypeName().str();
  std::vector<std::string> inputChanNames;
  for (auto op : node.commandArguments()) {
    inputChanNames.push_back(op.cast<mlir::StringAttr>().getValue().str());
  }
  std::vector<std::string> outputChanNames;
  for (auto op : node.commandResults()) {
    outputChanNames.push_back(op.cast<mlir::StringAttr>().getValue().str());
  }
  Signature signature = builder.getNodeTypeSignature(nodeTypeName,
                                                     inputChanNames,
                                                     outputChanNames);

  builder.startNode(signature, node.name().str());
  for (auto op : node.commandArguments()) {
    auto chanName = op.cast<mlir::StringAttr>().getValue().str();
    builder.addParam(chanName);
  }
  builder.startOutputs();
  for (auto op : node.commandResults()) {
    auto chanName = op.cast<mlir::StringAttr>().getValue().str();
    builder.addParam(chanName);
  }
  builder.endNode();
}

template <> void MLIRBuilder<mlir::hil::Nodes>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto nodeOp = mlir::cast<mlir::hil::Node>(op);
    MLIRBuilder::get(nodeOp, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Inst>::build() {
  builder.startInst(node.name().str());
  auto &operations = node.getBody()->getOperations();
  auto consOp = findElemByType<mlir::hil::Cons>(operations.begin(),
                                                operations.end());
  if (consOp) {
    MLIRBuilder::get(*consOp, builder).build();
  } else {
    std::cerr << "ERROR: `Cons` operator not found\n";
    exit(1);
  }
  builder.endInst();
}

template <> void MLIRBuilder<mlir::hil::Insts>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto instOp = mlir::cast<mlir::hil::Inst>(op);
    MLIRBuilder::get(instOp, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Graph>::build() {
  builder.startGraph(node.name().str());
  auto &operations = node.getBody()->getOperations();
  auto chansOp = findElemByType<mlir::hil::Chans>(operations.begin(),
                                                  operations.end());
  if (chansOp) {
    MLIRBuilder::get(*chansOp, builder).build();
  } else {
    std::cerr << "ERROR: `Chans` operator not found\n";
    exit(1);
  }
  auto nodesOp = findElemByType<mlir::hil::Nodes>(operations.begin(),
                                                  operations.end());
  if (nodesOp) {
    MLIRBuilder::get(*nodesOp, builder).build();
  } else {
    std::cerr << "ERROR: `Nodes` operator not found\n";
    exit(1);
  }
  auto instsOp = findElemByType<mlir::hil::Insts>(operations.begin(),
                                                  operations.end());
  if (instsOp) {
    MLIRBuilder::get(*instsOp, builder).build();
  }
  builder.endGraph();
}

template <> void MLIRBuilder<mlir::hil::Graphs>::build() {
  for (auto &operation : node.getBody()->getOperations()) {
    auto graphOp = mlir::cast<mlir::hil::Graph>(operation);
    MLIRBuilder::get(graphOp, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Model>::build() {
  builder.startModel(node.name().str());
  auto &operations = node.getBody()->getOperations();
  auto nodeTypesOp = findElemByType<mlir::hil::NodeTypes>(operations.begin(),
                                                          operations.end());
  if (nodeTypesOp) {
    MLIRBuilder::get(*nodeTypesOp, builder).build();
  } else {
    std::cerr << "ERROR: `NodeTypes` operator not found\n";
    exit(1);
  }
  auto graphsOp = findElemByType<mlir::hil::Graphs>(operations.begin(),
                                                    operations.end());
  if (graphsOp) {
    MLIRBuilder::get(*graphsOp, builder).build();
  } else {
    std::cerr << "ERROR: `Graphs` operator not found\n";
    exit(1);
  }
  builder.endModel();
}
} // namespace

namespace eda::hls::model {

std::shared_ptr<Model> parseModelFromMlir(const std::string &string) {
  auto &builder = Builder::get();
  auto mlirModelLayer = MLIRModule::loadFromMlir(string);
  auto model = MLIRBuilder<mlir::hil::Model>::buildModelFromMlir(
      mlirModelLayer, builder);
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