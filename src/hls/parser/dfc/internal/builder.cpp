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

  kernel->getWire(wire, Kernel::CREATE_ORIGINAL);
}

void Builder::connectWires(const ::dfc::wire *in, const ::dfc::wire *out) {
  assert(!kernels.empty() && "Wire connection outside a kernel");
  auto *kernel = kernels.back();

  auto *source = kernel->getWire(in,  Kernel::ACCESS_VERSION);
  auto *target = kernel->getWire(out, Kernel::CREATE_VERSION);

  kernel->in[target->name] = source;
  kernel->out[source->name].push_back(target);
}

void Builder::connectWires(const std::string &opcode,
                           const std::vector<const ::dfc::wire*> &in,
                           const std::vector<const ::dfc::wire*> &out) {
  assert(!kernels.empty() && "Wire connection outside a kernel");
  auto *kernel = kernels.back();

  // Create new inputs and connect them w/ the old ones.
  std::vector<Wire*> inputs;
  for (const auto *wire : in) {
    auto *source = kernel->getWire(wire, Kernel::ACCESS_VERSION);
    auto *target = kernel->getWire(wire, Kernel::CREATE_VERSION);

    kernel->in[target->name] = source;
    kernel->out[source->name].push_back(target);

    inputs.push_back(target);
  }

  // Compose outputs.
  std::vector<Wire*> outputs;
  for (const auto *wire : out) {
    auto *output = kernel->getWire(wire, Kernel::ACCESS_ORIGINAL);
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

  const bool create = mode == Kernel::CREATE_ORIGINAL;
  const bool access = mode == Kernel::ACCESS_ORIGINAL ||
                      mode == Kernel::ACCESS_VERSION;

  assert((!create || i == wires.end()) && "Wire already exists");
  assert((!access || i != wires.end()) && "Wire does not exist");

  if (mode == Kernel::ACCESS_ORIGINAL)
    return i->second;

  if (mode == Kernel::ACCESS_VERSION)
    return latest.find(wire->name)->second;

  const std::string name = (mode == Kernel::CREATE_VERSION) ?
        eda::utils::unique_name(wire->name) : wire->name;

  auto *result = new Wire(name,
                          wire->type(),
                          wire->direct == ::dfc::INPUT,
                          wire->direct == ::dfc::OUTPUT);

  wires.insert({ name, result });
  latest.insert({ wire->name, result });

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
  static unsigned id = 0;

  auto *nodetype = model->findNodetype(unit->fullName());

  if (nodetype == nullptr) {
    nodetype = createNodetype(unit, model);
    model->addNodetype(nodetype);
  }

  const std::string name = unit->opcode + std::to_string(id++);
  auto *node = new Node(name, *nodetype, *graph);

  for (auto *in : unit->in) {
    auto *input = graph->findChan(in->name);
    assert(input && "Input not found");

    input->target = { node, node->type.inputs[node->inputs.size()]};
    node->addInput(input);
  }

  for (auto *out: unit->out) {
    auto *output = graph->findChan(out->name);
    assert(output && "Output not found");

    output->source = { node, node->type.outputs[node->outputs.size()]};
    node->addOutput(output);
  }

  return node;
}

Graph* Builder::createGraph(const Kernel *kernel, Model *model) {
  auto *graph = new Graph(kernel->name, *model);

  // Create channels, sources, and sinks.
  for (const auto &[_, wire] : kernel->wires) {
    graph->addChan(createChan(wire, graph));

    const bool noInputs = kernel->in.find(wire->name) == kernel->in.end();
    const bool noOutputs = kernel->out.find(wire->name) == kernel->out.end();

    if (wire->input && noInputs && !noOutputs) {
      auto *source = new Unit("source", {}, { wire });
      graph->addNode(createNode(source, graph, model));
    } else if (wire->output && !noInputs && noOutputs) {
      auto *sink = new Unit("sink", { wire }, {});
      graph->addNode(createNode(sink, graph, model));
    }
  }

  // Create functional nodes and corresponding node types.
  for (auto *unit : kernel->units) {
    graph->addNode(createNode(unit, graph, model));
  }

  // Create duplication nodes and corresponding node types.
  for (const auto &[name, outputs] : kernel->out) {
    auto *input = kernel->getWire(name);
    auto *unit = new Unit("dup", { input }, outputs);
    graph->addNode(createNode(unit, graph, model));
  }

  return graph;
}

} // namespace eda::hls::parser::dfc
