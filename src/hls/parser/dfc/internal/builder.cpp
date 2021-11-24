//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/parser/dfc/internal/builder.h"

#include <cassert>
#include <sstream>

using namespace eda::hls::model;

namespace eda::hls::parser::dfc {

std::shared_ptr<Model> Builder::create(const std::string &name) {
  auto *model = new Model(name);

  for (const auto &kernel : kernels) {
    model->addGraph(createGraph(kernel, model));
  }

  return std::shared_ptr<Model>(model);
}

void Builder::startKernel(const std::string &name) {
  kernels.push_back(Kernel(name));
}

void Builder::declareWire(const ::dfc::wire *wire) {
  assert(!kernels.empty() && "Wire declaration outside a kernel");

  auto &kernel = kernels.back();
  kernel.wires.push_back(wire);
}

void Builder::connectWires(const ::dfc::wire *in, const ::dfc::wire *out) {
  assert(!kernels.empty() && "Wire connection outside a kernel");

  auto &kernel = kernels.back();
  kernel.fanout[in->name].push_back(out);
}

void Builder::connectWires(const std::string &opcode,
                           const std::vector<const ::dfc::wire*> &in,
                           const std::vector<const ::dfc::wire*> &out) {
  assert(!kernels.empty() && "Wire connection outside a kernel");

  // Create new inputs and connect them w/ the old ones.
  std::vector<const ::dfc::wire*> new_in(in.size());
  for (const auto *source : in) {
    const auto *target = source->new_wire();
    connectWires(source, target);
    new_in.push_back(target);
  }

  // Create a unit w/ the newly created inputs.
  auto &kernel = kernels.back();
  kernel.units.push_back(Unit(opcode, new_in, out));
}

std::string Builder::Unit::fullName() const {
  std::stringstream fullname;

  fullname << opcode << "_" << in.size() << "x" << out.size();
  for (const auto *wire : in)
    fullname << "_" << wire->type();
  for (const auto *wire : out)
    fullname << "_" << wire->type();

  std::string result = fullname.str();
  std::replace_if(result.begin(), result.end(),
    [](char c) { return c == '<' || c == '>'; }, '_');

  return result;
}

Port* Builder::createPort(const ::dfc::wire *wire, unsigned latency) {
  return new Port(wire->name,   // Name
                  wire->type(), // Type
                  1.0,          // Flow
                  latency,      // Latency
                  false,        // Constant
                  0);           // Value 
}

NodeType* Builder::createNodetype(const Unit &unit, Model *model) {
  NodeType *nodetype = new NodeType(unit.fullName(), *model);
  for (const auto *in : unit.in)
    nodetype->addInput(createPort(in, 0));
  for (const auto *out : unit.out)
    nodetype->addOutput(createPort(out, 1));

  return nodetype;
}

Chan* Builder::createChan(const ::dfc::wire *wire, Graph *graph) {
  return new Chan(wire->name, wire->type(), *graph);
}

Node* Builder::createNode(const Unit &unit, Graph *graph, Model *model) {
  NodeType *nodetype = createNodetype(unit, model);
  model->addNodetype(nodetype);

  Node *node = new Node(unit.opcode, *nodetype, *graph);
  for (const auto *in : unit.in)
    node->addInput(graph->findChan(in->name));
  for (const auto *out: unit.out)
    node->addOutput(graph->findChan(out->name));

  return node;
}

Graph* Builder::createGraph(const Kernel &kernel, Model *model) {
  Graph *graph = new Graph(kernel.name, *model);

  // Create channels.
  for (const auto *wire : kernel.wires) {
    graph->addChan(createChan(wire, graph));
  }

  // Create functional nodes and corresponding node types.
  for (const auto &unit : kernel.units) {
    graph->addNode(createNode(unit, graph, model));
  }

  // Create duplication nodes and corresponding node types.
  for (const auto &entry : kernel.fanout) {
    const std::string &name = entry.first;
    const auto i = std::find_if(kernel.wires.begin(), kernel.wires.end(),
      [&name](const ::dfc::wire *wire) { return wire->name == name; });
    Unit unit("dup", { *i }, entry.second);
    graph->addNode(createNode(unit, graph, model));
  }

  return graph;
}

} // namespace eda::hls::parser::dfc
