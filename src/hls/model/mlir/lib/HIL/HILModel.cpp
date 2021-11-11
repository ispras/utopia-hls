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

#include "HIL/HILDialect.h"
#include "HIL/HILModel.h"
#include "HIL/HILOps.h"
#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"

using eda::hls::model::Model;
using eda::hls::parser::hil::Builder;

namespace detail {
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
T find_only_elem_by_type(Iterator first, Iterator last) {
  auto v = find_elems_by_type<T>(first, last);
  assert(v.size() == 1);
  return v[0];
}

template <typename T> class MLIRBuilder {
public:
  MLIRBuilder(T &node, Builder &builder) : node(node), builder(builder) {}
  void build();
  template <typename U> static MLIRBuilder<U> get(U &node, Builder &builder) {
    return MLIRBuilder<U>(node, builder);
  }

private:
  T &node;
  Builder &builder;
};

template <> void MLIRBuilder<mlir::hil::InputArgType>::build() {
  builder.addPort(node.getName(), node.getTypeName(),
                  std::to_string(*node.getFlow()),
                  "0");
}

template <> void MLIRBuilder<mlir::hil::OutputArgType>::build() {
  builder.addPort(node.getName(), node.getTypeName(),
                  std::to_string(*node.getFlow()),
                  std::to_string(node.getLatency()),
                  node.getValue());
}

template <> void MLIRBuilder<mlir::hil::NodeType>::build() {
  builder.startNodetype(node.name().str());
  // Build inputs
  for (auto op : node.commandArguments()) {
    auto in_port_op =
        op.cast<mlir::TypeAttr>().getValue().cast<mlir::hil::InputArgType>();
    MLIRBuilder::get(in_port_op, builder).build();
  }
  // Build outputs
  builder.startOutputs();
  for (auto op : node.commandResults()) {
    auto out_port_op =
        op.cast<mlir::TypeAttr>().getValue().cast<mlir::hil::OutputArgType>();
    MLIRBuilder::get(out_port_op, builder).build();
  }
  builder.endNodetype();
}

template <> void MLIRBuilder<mlir::hil::NodeTypes>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto nodetype_op = mlir::cast<mlir::hil::NodeType>(op);
    MLIRBuilder::get(nodetype_op, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Chan>::build() {
  builder.addChan(node.typeName().str(), node.varName().str());
}

template <> void MLIRBuilder<mlir::hil::Chans>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto chan_op = mlir::cast<mlir::hil::Chan>(op);
    MLIRBuilder::get(chan_op, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Node>::build() {
  builder.startNode(node.nodeTypeName().str(), node.name().str());
  for (auto op : node.commandArguments()) {
    auto chan_name = op.cast<mlir::StringAttr>().getValue().str();
    builder.addParam(chan_name);
  }
  builder.startOutputs();
  for (auto op : node.commandResults()) {
    auto chan_name = op.cast<mlir::StringAttr>().getValue().str();
    builder.addParam(chan_name);
  }
  builder.endNode();
}

template <> void MLIRBuilder<mlir::hil::Nodes>::build() {
  for (auto &op : node.getBody()->getOperations()) {
    auto node_op = mlir::cast<mlir::hil::Node>(op);
    MLIRBuilder::get(node_op, builder).build();
  }
}

template <> void MLIRBuilder<mlir::hil::Graph>::build() {
  builder.startGraph(node.name().str());
  auto &ops = node.getBody()->getOperations();
  auto chans_op =
      find_only_elem_by_type<mlir::hil::Chans>(ops.begin(), ops.end());
  MLIRBuilder::get(chans_op, builder).build();
  auto nodes_op =
      find_only_elem_by_type<mlir::hil::Nodes>(ops.begin(), ops.end());
  MLIRBuilder::get(nodes_op, builder).build();
  builder.endGraph();
}

template <> void MLIRBuilder<mlir::hil::Model>::build() {
  builder.startModel(node.name().str());
  auto &ops = node.getBody()->getOperations();
  auto nodetypes_op =
      find_only_elem_by_type<mlir::hil::NodeTypes>(ops.begin(), ops.end());
  MLIRBuilder::get(nodetypes_op, builder).build();
  for (auto graph_op :
       find_elems_by_type<mlir::hil::Graph>(ops.begin(), ops.end())) {
    MLIRBuilder::get(graph_op, builder).build();
  }
  builder.endModel();
}

class MLIRModelLayer {
public:
  static MLIRModelLayer load_from_mlir(const std::string &s) {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<mlir::hil::HILDialect>();
    context->getOrLoadDialect<mlir::StandardOpsDialect>();
    auto module = mlir::parseSourceString(s, context.get());
    return MLIRModelLayer(std::move(context), std::move(module));
  }

  void traverse_with_builder(Builder &builder) {
    auto op = mlir::cast<mlir::hil::Model>(*module->getOps().begin());
    MLIRBuilder(op, builder).build();
  }

  MLIRModelLayer(MLIRModelLayer &&oth)
      : context(std::move(oth.context)), module(std::move(oth.module)),
        pm(std::move(oth.pm)) {}

private:
  MLIRModelLayer(std::unique_ptr<mlir::MLIRContext> &&context,
                 mlir::OwningModuleRef &&module)
      : context(std::move(context)), module(std::move(module)),
        pm(std::make_unique<mlir::PassManager>(this->context.get())) {}
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningModuleRef module;
  std::unique_ptr<mlir::PassManager> pm;
};
} // namespace detail

namespace eda::hls::model {

std::unique_ptr<Model> parse_model_from_mlir(const std::string &s) {
  auto &builder = Builder::get();
  auto mlir_model_layer = detail::MLIRModelLayer::load_from_mlir(s);
  mlir_model_layer.traverse_with_builder(builder);
  auto m = builder.create();
  std::cerr << *m << std::endl;
  return std::unique_ptr<Model>(std::move(m));
}

std::unique_ptr<Model> parse_model_from_mlir_file(const std::string &filename) {
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

std::string dump_model_to_mlir(Model &m) { return ""; } // TODO
} // namespace eda::hls::model
