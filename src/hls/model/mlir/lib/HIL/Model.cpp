//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
#include "mlir/InitAllDialects.h"

#include <fstream>
#include <iostream>
#include <optional>

#include "HIL/Dialect.h"
#include "HIL/Dumper.h"
#include "HIL/Model.h"
#include "HIL/Ops.h"
#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"

using eda::hls::model::Model;
using eda::hls::parser::hil::Builder;

namespace detail {
class MLIRModule {
public:
  static MLIRModule load_from_mlir(const std::string &s) {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<mlir::hil::HILDialect>();
    context->getOrLoadDialect<mlir::StandardOpsDialect>();
    auto module = mlir::parseSourceString(s, context.get());
    return MLIRModule(std::move(context), std::move(module));
  }

  mlir::hil::Model get_root() {
    return mlir::cast<mlir::hil::Model>(*module->getOps().begin());
  }

  MLIRModule(MLIRModule &&oth)
      : context(std::move(oth.context)), module(std::move(oth.module)),
        pm(std::move(oth.pm)) {}

private:
  MLIRModule(std::unique_ptr<mlir::MLIRContext> &&context,
             mlir::OwningModuleRef &&module)
      : context(std::move(context)), module(std::move(module)),
        pm(std::make_unique<mlir::PassManager>(this->context.get())) {}
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningModuleRef module;
  std::unique_ptr<mlir::PassManager> pm;
};

template <typename T, typename Iterator>
std::vector<T> find_elems_by_type(Iterator first, Iterator last) {
  std::vector<T> result;
  for (; first != last; ++first) {
    auto elem = mlir::dyn_cast<T>(*first);
    if (elem) {
      result.push_back(elem);
    }
  }
  return result;
}

template <typename T, typename Iterator>
std::optional<T> find_elem_by_type(Iterator first, Iterator last) {
  for (; first != last; ++first) {
    auto elem = mlir::dyn_cast<T>(*first);
    if (elem) {
      return elem;
    }
  }
  return std::nullopt;
}

template <typename T>
class MLIRBuilder {
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

template <>
void MLIRBuilder<mlir::hil::InputArgType>::build() {
  builder_.addPort(node_.getName(), node_.getTypeName(),
                   std::to_string(*node_.getFlow()), "0");
}

template <>
void MLIRBuilder<mlir::hil::OutputArgType>::build() {
  builder_.addPort(node_.getName(), node_.getTypeName(),
                   std::to_string(*node_.getFlow()),
                   std::to_string(node_.getLatency()), node_.getValue());
}

template <>
void MLIRBuilder<mlir::hil::NodeType>::build() {
  builder_.startNodetype(node_.name().str());
  // Build inputs
  for (auto op : node_.commandArguments()) {
    auto in_port_op =
        op.cast<mlir::TypeAttr>().getValue().cast<mlir::hil::InputArgType>();
    MLIRBuilder::get(in_port_op, builder_).build();
  }
  // Build outputs
  builder_.startOutputs();
  for (auto op : node_.commandResults()) {
    auto out_port_op =
        op.cast<mlir::TypeAttr>().getValue().cast<mlir::hil::OutputArgType>();
    MLIRBuilder::get(out_port_op, builder_).build();
  }
  builder_.endNodetype();
}

template <>
void MLIRBuilder<mlir::hil::NodeTypes>::build() {
  for (auto &op : node_.getBody()->getOperations()) {
    auto nodetype_op = mlir::cast<mlir::hil::NodeType>(op);
    MLIRBuilder::get(nodetype_op, builder_).build();
  }
}

template <>
void MLIRBuilder<mlir::hil::Chan>::build() {
  builder_.addChan(node_.typeName().str(), node_.varName().str());
}

template <>
void MLIRBuilder<mlir::hil::Chans>::build() {
  for (auto &op : node_.getBody()->getOperations()) {
    auto chan_op = mlir::cast<mlir::hil::Chan>(op);
    MLIRBuilder::get(chan_op, builder_).build();
  }
}

template <>
void MLIRBuilder<mlir::hil::Node>::build() {
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

template <>
void MLIRBuilder<mlir::hil::Nodes>::build() {
  for (auto &op : node_.getBody()->getOperations()) {
    auto node_op = mlir::cast<mlir::hil::Node>(op);
    MLIRBuilder::get(node_op, builder_).build();
  }
}

template <>
void MLIRBuilder<mlir::hil::Graph>::build() {
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

template <>
void MLIRBuilder<mlir::hil::Model>::build() {
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
} // namespace detail

namespace eda::hls::model {

std::shared_ptr<Model> parse_model_from_mlir(const std::string &s) {
  auto &builder = Builder::get();
  auto mlir_model_layer = detail::MLIRModule::load_from_mlir(s);
  auto m = detail::MLIRBuilder<mlir::hil::Model>::build_model_from_mlir(
      mlir_model_layer, builder);
  std::cerr << "***********MODEL_BEGIN**********" << std::endl;
  std::cerr << *m << std::endl;
  std::cerr << "************MODEL_END***********" << std::endl;
  std::cerr << "***********MLIR_BEGIN***********" << std::endl;
  dump_model_mlir(*m, std::cerr);
  std::cerr << "************MLIR_END************" << std::endl;
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
  std::cout << buf.str() << std::endl;
  return parse_model_from_mlir(buf.str());
}

} // namespace eda::hls::model
