//===- Model.cpp - MLIR model ---------------*- C++ -*---------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"

#include <fstream>
#include <iostream>
#include <optional>

#include "HIL/Dialect.h"
#include "HIL/Dumper.h"
#include "HIL/Model.h"
#include "HIL/Ops.h"
#include "HIL/Utils.h"
#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"

using eda::hls::model::Model;
using eda::hls::parser::hil::Builder;

namespace mlir::model {
MLIRModule MLIRModule::load_from_mlir(const std::string &s) {
  auto context = std::make_unique<mlir::MLIRContext>();
  context->getOrLoadDialect<mlir::hil::HILDialect>();
  context->getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(s, context.get());
  return {std::move(context), std::move(module)};
}

MLIRModule MLIRModule::load_from_mlir_file(const std::string &filename) {
  std::ifstream file{filename};
  std::stringstream buf;
  if (file) {
    buf << file.rdbuf();
  } else { // fallback to stdin
    buf << std::cin.rdbuf();
  }
  return load_from_mlir(buf.str());
}

MLIRModule MLIRModule::load_from_model(const eda::hls::model::Model &m) {
  std::stringstream buf;
  dump_model_mlir(m, buf);
  return load_from_mlir(buf.str());
}

void MLIRModule::print(llvm::raw_ostream &os) { module_->print(os); }

mlir::hil::Model MLIRModule::get_root() {
  return mlir::cast<mlir::hil::Model>(*module_->getOps().begin());
}

MLIRModule::MLIRModule(MLIRModule &&oth)
    : context_(std::move(oth.context_)), module_(std::move(oth.module_)) {}

MLIRModule &MLIRModule::operator=(MLIRModule &&oth) {
  module_ = std::move(oth.module_);
  context_ = std::move(oth.context_);
  return *this;
}

MLIRModule MLIRModule::clone() { return {context_, module_->clone()}; }

MLIRContext *MLIRModule::get_context() { return module_->getContext(); }

MLIRModule::MLIRModule(std::shared_ptr<mlir::MLIRContext> context,
                       mlir::OwningOpRef<mlir::ModuleOp> &&module)
    : context_(std::move(context)), module_(std::move(module)) {}
} // namespace mlir::model

namespace {
using mlir::model::MLIRModule;

template <typename T> class MLIRBuilder {
public:
  MLIRBuilder(T &node, Builder &builder) : node_(node), builder_(builder) {}
  MLIRBuilder(T &&node, Builder &builder) : node_(node), builder_(builder) {}
  void build();
  template <typename U>
  static MLIRBuilder<std::decay_t<U>> get(U &&node, Builder &builder) {
    return MLIRBuilder<std::decay_t<U>>(std::forward<U>(node), builder);
  }
  static std::shared_ptr<Model> build_model_from_mlir(MLIRModule &mlir_model,
                                                      Builder &builder);

private:
  T &node_;
  Builder &builder_;
};

template <typename T>
std::shared_ptr<Model>
MLIRBuilder<T>::build_model_from_mlir(MLIRModule &mlir_model,
                                      Builder &builder) {
  MLIRBuilder(mlir_model.get_root(), builder).build();
  return builder.create();
}

template <> void MLIRBuilder<mlir::hil::InputPortAttr>::build() {
  builder_.addPort(node_.getName(), node_.getTypeName(),
                   std::to_string(*node_.getFlow()), "0");
}

template <> void MLIRBuilder<mlir::hil::OutputPortAttr>::build() {
  builder_.addPort(node_.getName(), node_.getTypeName(),
                   std::to_string(*node_.getFlow()),
                   std::to_string(node_.getLatency()), node_.getValue());
}

template <> void MLIRBuilder<mlir::hil::NodeType>::build() {
  builder_.startNodetype(node_.name().str());
  // Build inputs
  for (auto op : node_.commandArguments()) {
    auto in_port_op =
        op.cast<mlir::Attribute>().cast<mlir::hil::InputPortAttr>();
    MLIRBuilder::get(in_port_op, builder_).build();
  }
  // Build outputs
  builder_.startOutputs();
  for (auto op : node_.commandResults()) {
    auto out_port_op =
        op.cast<mlir::Attribute>().cast<mlir::hil::OutputPortAttr>();
    MLIRBuilder::get(out_port_op, builder_).build();
  }
  builder_.endNodetype();
}

template <> void MLIRBuilder<mlir::hil::NodeTypes>::build() {
  for (auto &op : node_.getBody()->getOperations()) {
    auto nodetype_op = mlir::cast<mlir::hil::NodeType>(op);
    MLIRBuilder::get(nodetype_op, builder_).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Chan>::build() {
  builder_.addChan(node_.typeName().str(), node_.varName().str());
}

template <> void MLIRBuilder<mlir::hil::Chans>::build() {
  for (auto &op : node_.getBody()->getOperations()) {
    auto chan_op = mlir::cast<mlir::hil::Chan>(op);
    MLIRBuilder::get(chan_op, builder_).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Node>::build() {
  builder_.startNode(node_.nodeTypeName().str(), node_.name().str());
  for (auto op : node_.commandArguments()) {
    auto chan_name = op.cast<mlir::StringAttr>().getValue().str();
    builder_.addParam(chan_name);
  }
  builder_.startOutputs();
  for (auto op : node_.commandResults()) {
    auto chan_name = op.cast<mlir::StringAttr>().getValue().str();
    builder_.addParam(chan_name);
  }
  builder_.endNode();
}

template <> void MLIRBuilder<mlir::hil::Nodes>::build() {
  for (auto &op : node_.getBody()->getOperations()) {
    auto node_op = mlir::cast<mlir::hil::Node>(op);
    MLIRBuilder::get(node_op, builder_).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Graph>::build() {
  builder_.startGraph(node_.name().str());
  auto &ops = node_.getBody()->getOperations();
  auto chans_op = find_elem_by_type<mlir::hil::Chans>(ops.begin(), ops.end());
  if (chans_op) {
    MLIRBuilder::get(*chans_op, builder_).build();
  } else {
    std::cerr << "ERROR: `Chans` operator not found\n";
    exit(1);
  }
  auto nodes_op = find_elem_by_type<mlir::hil::Nodes>(ops.begin(), ops.end());
  if (nodes_op) {
    MLIRBuilder::get(*nodes_op, builder_).build();
  } else {
    std::cerr << "ERROR: `Nodes` operator not found\n";
    exit(1);
  }
  builder_.endGraph();
}

template <> void MLIRBuilder<mlir::hil::Model>::build() {
  builder_.startModel(node_.name().str());
  auto &ops = node_.getBody()->getOperations();
  auto nodetypes_op =
      find_elem_by_type<mlir::hil::NodeTypes>(ops.begin(), ops.end());
  if (nodetypes_op) {
    MLIRBuilder::get(*nodetypes_op, builder_).build();
  } else {
    std::cerr << "ERROR: `NodeTypes` operator not found\n";
    exit(1);
  }
  for (auto graph_op :
       find_elems_by_type<mlir::hil::Graph>(ops.begin(), ops.end())) {
    MLIRBuilder::get(graph_op, builder_).build();
  }
  builder_.endModel();
}
} // namespace

namespace eda::hls::model {

std::shared_ptr<Model> parse_model_from_mlir(const std::string &s) {
  auto &builder = Builder::get();
  auto mlir_model_layer = MLIRModule::load_from_mlir(s);
  auto m = MLIRBuilder<mlir::hil::Model>::build_model_from_mlir(
      mlir_model_layer, builder);
  /* std::cerr << "***********MODEL_BEGIN**********" << std::endl; */
  /* std::cerr << *m << std::endl; */
  /* std::cerr << "************MODEL_END***********" << std::endl; */
  /* std::cerr << "***********MLIR_BEGIN***********" << std::endl; */
  /* dump_model_mlir(*m, std::cerr); */
  /* std::cerr << "************MLIR_END************" << std::endl; */
  return std::shared_ptr<Model>(std::move(m));
}

std::shared_ptr<Model> parse_model_from_mlir_file(const std::string &filename) {
  std::ifstream file{filename};
  std::stringstream buf;
  if (file) {
    buf << file.rdbuf();
  } else { // fallback to stdin
    buf << std::cin.rdbuf();
  }
  //std::cout << buf.str() << std::endl;
  return parse_model_from_mlir(buf.str());
}

} // namespace eda::hls::model
