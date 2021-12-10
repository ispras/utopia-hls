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
#include <iostream>
#include <sstream>

using namespace eda::hls::model;

namespace eda::hls::parser::dfc {

std::shared_ptr<Model> Builder::create(const std::string &name) {
  auto *model = new Model(name);

  for (auto *kernel : kernels) {
    kernel->reduce();
    model->addGraph(getGraph(kernel, model));
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

  kernel->in[target->name] = source->name;
  kernel->out[source->name].push_back(target->name);
}

void Builder::connectWires(const std::string &opcode,
                           const std::vector<const ::dfc::wire*> &in,
                           const std::vector<const ::dfc::wire*> &out) {
  assert(!kernels.empty() && "Wire connection outside a kernel");
  auto *kernel = kernels.back();

  // Create new inputs and connect them w/ the old ones.
  std::vector<std::string> inputs;
  for (const auto *wire : in) {
    auto *source = kernel->getWire(wire, Kernel::ACCESS_VERSION);
    auto *target = kernel->getWire(wire, Kernel::CREATE_VERSION);

    kernel->in[target->name] = source->name;
    kernel->out[source->name].push_back(target->name);

    inputs.push_back(target->name);
  }

  // Compose outputs.
  std::vector<std::string> outputs;
  for (const auto *wire : out) {
    auto *output = kernel->getWire(wire, Kernel::ACCESS_ORIGINAL);
    outputs.push_back(output->name);
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
  auto i = originals.find(name);

  if (i == originals.end())
    i = replaced.find(name);

  return i->second;
}

Builder::Wire* Builder::Kernel::getWire(const ::dfc::wire *wire, Mode mode) {
  auto i = originals.find(wire->name);

  const bool access = mode == Kernel::ACCESS_ORIGINAL ||
                      mode == Kernel::ACCESS_VERSION;

  assert((!access || i != originals.end()) && "Wire does not exist");

  if ((mode == Kernel::ACCESS_ORIGINAL) ||
      (mode == Kernel::CREATE_ORIGINAL && i != originals.end()))
    return i->second;

  if (mode == Kernel::ACCESS_VERSION)
    return versions.find(wire->name)->second;

  const std::string name = (mode == Kernel::CREATE_VERSION) ?
        eda::utils::unique_name(wire->name) : wire->name;

  auto *result = new Wire(name,
                          wire->type(),
                          wire->direct != ::dfc::OUTPUT,
                          wire->direct != ::dfc::INPUT);

  wires.push_back(result);
  versions[wire->name] = originals[name] = result;

  return result;
}

void Builder::Kernel::reduce() {
  for (auto *wire : wires) {
    const auto input = wire->name;
    const auto i = out.find(input);

    if (i == out.end() || i->second.size() != 1)
      continue;

    const auto output = i->second.front();
    const auto j = out.find(output);

    if (j == out.end())
      out.erase(input);
    else {
      out[input] = j->second;
      for (const auto &next : j->second)
        in[next] = input;
    }

    in.erase(output);
    out.erase(output);

    originals.erase(output);
    versions.erase(output);

    replaced[output] = wire;
  }
}

Port* Builder::getPort(const Wire *wire, unsigned latency) {
  return new Port(wire->name, // Name
                  wire->type, // Type
                  1.0,        // Flow
                  latency,    // Latency
                  false,      // Constant
                  0);         // Value 
}

Chan* Builder::getChan(const Wire *wire, Graph *graph) {
  auto *chan = graph->findChan(wire->name);

  if (!chan) {
    chan = new Chan(wire->name, wire->type, *graph);
    graph->addChan(chan);
  }

  return chan; 
}

NodeType* Builder::getNodetype(const Kernel *kernel,
                               const Unit *unit,
                               Model *model) {
  auto *nodetype = new NodeType(unit->fullName(), *model);

  for (const auto &name : unit->in) {
    auto *wire = kernel->getWire(name);
    nodetype->addInput(getPort(wire, 0));
  }

  for (const auto &name : unit->out) {
    auto *wire = kernel->getWire(name);
    nodetype->addOutput(getPort(wire, 1));
  }

  return nodetype;
}

Node* Builder::getNode(const Kernel *kernel,
                       const Unit *unit,
                       Graph *graph,
                       Model *model) {
  static unsigned id = 0;

  auto *nodetype = model->findNodetype(unit->fullName());
  if (!nodetype) {
    nodetype = getNodetype(kernel, unit, model);
    model->addNodetype(nodetype);
  }

  const std::string nodeName = unit->opcode + std::to_string(id++);
  auto *node = new Node(nodeName, *nodetype, *graph);

  for (const auto &name : unit->in) {
    auto *wire = kernel->getWire(name);
    auto *input = getChan(wire, graph);
    assert(input && "Input not found");

    input->target = { node, node->type.inputs[node->inputs.size()] };
    node->addInput(input);
  }

  for (const auto &name : unit->out) {
    auto *wire = kernel->getWire(name);
    auto *output = getChan(wire, graph);
    assert(output && "Output not found");

    output->source = { node, node->type.outputs[node->outputs.size()] };
    node->addOutput(output);
  }

  return node;
}

Graph* Builder::getGraph(const Kernel *kernel, Model *model) {
  auto *graph = new Graph(kernel->name, *model);

  // Create sources, and sinks.
  for (const auto &[_, wire] : kernel->originals) {

    const bool noInputs = kernel->in.find(wire->name) == kernel->in.end();
    const bool noOutputs = kernel->out.find(wire->name) == kernel->out.end();

    if (wire->input && noInputs && !noOutputs) {
      auto *source = new Unit("source", {}, { wire->name });
      graph->addNode(getNode(kernel, source, graph, model));
    } else if (wire->output && !noInputs && noOutputs) {
      auto *sink = new Unit("sink", { wire->name }, {});
      graph->addNode(getNode(kernel, sink, graph, model));
    }
  }

  // Create functional nodes and corresponding node types.
  for (auto *unit : kernel->units) {
    graph->addNode(getNode(kernel, unit, graph, model));
  }

  // Create duplication nodes and corresponding node types.
  for (const auto &[input, outputs] : kernel->out) {
    auto *unit = new Unit("dup", { input }, outputs);
    graph->addNode(getNode(kernel, unit, graph, model));
  }

  return graph;
}

} // namespace eda::hls::parser::dfc
