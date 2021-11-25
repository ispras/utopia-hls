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

  for (const auto *kernel : kernels) {
    model->addGraph(createGraph(kernel, model));
  }

  return std::shared_ptr<Model>(model);
}

void Builder::startKernel(const std::string &name) {
  kernels.push_back(new Kernel(name));
}

void Builder::declareWire(const ::dfc::wire *wire) {
  assert(!kernels.empty() && "Wire declaration outside a kernel");
  auto *kernel = kernels.back();

  kernel->getWire(wire, Kernel::CREATE);
}

void Builder::connectWires(const ::dfc::wire *in, const ::dfc::wire *out) {
  assert(!kernels.empty() && "Wire connection outside a kernel");
  auto *kernel = kernels.back();

  auto *source = kernel->getWire(in, Kernel::ACCESS);
  auto *target = kernel->getWire(out, Kernel::CREATE_IF_NOT_EXISTS);

  kernel->fanout[source->name].push_back(target);
}

void Builder::connectWires(const std::string &opcode,
                           const std::vector<const ::dfc::wire*> &in,
                           const std::vector<const ::dfc::wire*> &out) {
  assert(!kernels.empty() && "Wire connection outside a kernel");
  auto *kernel = kernels.back();

  // Create new inputs and connect them w/ the old ones.
  std::vector<Wire*> inputs;
  for (const auto *wire : in) {
    auto *source = kernel->getWire(wire, Kernel::ACCESS);
    auto *target = kernel->getWire(wire, Kernel::CREATE_COPY);

    kernel->fanout[source->name].push_back(target);
    inputs.push_back(target);
  }

  // Compose outputs.
  std::vector<Wire*> outputs;
  for (const auto *wire : out) {
    auto *output = kernel->getWire(wire, Kernel::ACCESS);
    outputs.push_back(output);
  }

  // Create a unit w/ the newly created inputs.
  kernel->units.push_back(new Unit(opcode, inputs, outputs));
}

std::string Builder::Unit::fullName() const {
  std::stringstream fullname;

  fullname << opcode << "_" << in.size() << "x" << out.size();
  /*
  for (auto *wire : in)
    fullname << "_" << wire->type;
  for (auto *wire : out)
    fullname << "_" << wire->type;
  */
  auto result = fullname.str();
  std::replace_if(result.begin(), result.end(),
    [](char c) { return c == '<' || c == '>' || c == ','; }, '_');

  return result;
}

Builder::Wire* Builder::Kernel::getWire(const std::string &name) const {
  auto i = wires.find(name);
  return i != wires.end() ? i->second : nullptr;
}

Builder::Wire* Builder::Kernel::getWire(const ::dfc::wire *wire, Mode mode) {
  auto i = wires.find(wire->name);

  assert((mode != Kernel::CREATE || i == wires.end()) && "Wire already exists");
  assert((mode != Kernel::ACCESS || i != wires.end()) && "Wire does not exist");

  if (i != wires.end() && mode != Kernel::CREATE_COPY)
    return i->second;

  const std::string name = (mode == Kernel::CREATE_COPY)
      ? eda::utils::unique_name(wire->name)
      : wire->name;

  auto *result = new Wire(name, wire->type());
  wires.insert({ name, result });

  return result;
}

Port* Builder::createPort(const Wire *wire, unsigned latency) {
  return new Port(wire->name, // Name
                  wire->type, // Type
                  1.0,        // Flow
                  latency,    // Latency
                  false,      // Constant
                  0);         // Value 
}

NodeType* Builder::createNodetype(const Unit *unit, Model *model) {
  auto *nodetype = new NodeType(unit->fullName(), *model);
  for (auto *in : unit->in)
    nodetype->addInput(createPort(in, 0));
  for (auto *out : unit->out)
    nodetype->addOutput(createPort(out, 1));

  return nodetype;
}

Chan* Builder::createChan(const Wire *wire, Graph *graph) {
  return new Chan(wire->name, wire->type, *graph);
}

Node* Builder::createNode(const Unit *unit, Graph *graph, Model *model) {
  auto *nodetype = createNodetype(unit, model);
  model->addNodetype(nodetype);

  auto *node = new Node(unit->opcode, *nodetype, *graph);

  for (auto *in : unit->in) {
    auto *input = graph->findChan(in->name);
    input->target = { node, node->type.inputs[node->inputs.size()]};
    node->addInput(input);
  }

  for (auto *out: unit->out) {
    auto *output = graph->findChan(out->name);
    output->source = { node, node->type.outputs[node->outputs.size()]};
    node->addOutput(output);
  }

  return node;
}

Graph* Builder::createGraph(const Kernel *kernel, Model *model) {
  auto *graph = new Graph(kernel->name, *model);

  // Create channels.
  for (const auto &[_, wire] : kernel->wires) {
    graph->addChan(createChan(wire, graph));
  }

  // Create functional nodes and corresponding node types.
  for (auto *unit : kernel->units) {
    graph->addNode(createNode(unit, graph, model));
  }

  // Create duplication nodes and corresponding node types.
  for (const auto &[name, outputs] : kernel->fanout) {
    auto *input = kernel->getWire(name);
    auto *unit = new Unit("dup", { input }, outputs);
    graph->addNode(createNode(unit, graph, model));
  }

  return graph;
}

} // namespace eda::hls::parser::dfc
