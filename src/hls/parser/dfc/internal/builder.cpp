//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/parser/dfc/internal/builder.h"
#include "hls/model/model.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <unordered_set>

using Con = eda::hls::model::Con;
using Graph = eda::hls::model::Graph;
using Model = eda::hls::model::Model;
using Node = eda::hls::model::Node;
using Port = eda::hls::model::Port;
using Signature = eda::hls::model::Signature;

namespace eda::hls::parser::dfc {

//===----------------------------------------------------------------------===//
// Unit
//===----------------------------------------------------------------------===//

std::string Builder::Unit::getFullName() const {
  std::stringstream fullName;

  fullName << opcode << "_" << in.size() << "x" << out.size();

  // Temporal solution.
  if (opcode == "const") {
    fullName << "_" << out.back()->value;
  }
  if (opcode == "INSTANCE") {
    fullName << "_" << instance->kernelName;
  }
  //----------------------------------------------------------------------------

  auto result = fullName.str();

  std::replace_if(result.begin(), result.end(),
    [](char c) { return c == '<' || c == '>' || c == ','; }, '_');
  
  return result;
}

//===----------------------------------------------------------------------===//
// Kernel
//===----------------------------------------------------------------------===//

Builder::Wire *Builder::Kernel::getWire(const ::dfc::wire *wire, Mode mode) {
  auto i = originals.find(wire->name);

  const bool access = (mode == Kernel::ACCESS_ORIGINAL) ||
                      (mode == Kernel::ACCESS_VERSION);

  assert((!access || i != originals.end()) && "Wire does not exist!\n");

  if ((mode == Kernel::ACCESS_ORIGINAL) ||
      (mode == Kernel::CREATE_ORIGINAL && i != originals.end()))
    return i->second;

  if (mode == Kernel::ACCESS_VERSION)
    return versions.find(wire->name)->second;

  const std::string name = (mode == Kernel::CREATE_VERSION)
      ? eda::utils::unique_name(wire->name)
      : wire->name;

  const auto type = wire->type();

  const bool isInput  = (wire->direct != ::dfc::OUTPUT);
  const bool isOutput = (wire->direct != ::dfc::INPUT);
  const bool isConst  = (wire->kind == ::dfc::CONST);

  const std::string value = wire->value.to_string();

  auto *result = new Wire(name, type, isInput, isOutput, isConst, value);

  wires.push_back(result);
  versions[wire->name] = originals[name] = result;

  return result;
}

Builder::Unit *Builder::Kernel::getUnit(const std::string &opcode,
                                        const std::vector<Wire*> &in,
                                        const std::vector<Wire*> &out) {
  auto *unit = new Unit(opcode, in, out);
  units.push_back(unit);

  return unit;
}

void Builder::Kernel::createInstanceUnit(const std::string &opcode,
                                         const std::string &instanceName,
                                         const std::string &kernelName) {
  auto *unit = new Unit(opcode, instanceName, kernelName);
  units.push_back(unit);
  instanceUnits.insert({instanceName, unit});
}

void Builder::Kernel::connect(Wire *source, Wire *target) {
  auto *consumer = source->consumer;
  auto *producer = source->producer;

  if (!consumer) {
    getUnit("dup", { source }, { target });
  } else if (consumer->opcode == "dup") {
    consumer->addOutput(target);
  } else if (producer && producer->opcode == "dup") {
    producer->addOutput(target);
  } else {
    assert(false && "Duplication unit expected!\n");
  }
}

void Builder::Kernel::transform() {
  if (isTransformed)
    return;

  // Remove redundant units.
  std::unordered_set<Unit*> removing;

  for (auto *unit : units) {
    if (unit->opcode != "dup" ||
        unit->in.size() != 1 ||
        unit->out.size() != 1) {
      continue;
    }

    auto *input = unit->in.front();
    auto *output = unit->out.front();

    auto *next = output->consumer;
    input->consumer = next;

    if (next) {
      // When we delete some wire we need to modify the bindings.
      if (next->opcode == "INSTANCE") {
        auto source = next->instance->bindingsToInputs[output];
        next->instance->modifyBindingToInput(output, input, source);
      }
      //------------------------------------------------------------------------
      std::replace(next->in.begin(), next->in.end(), output, input);
      originals.erase(output->name);
      versions.erase(output->name);
    } else {
      auto *previous = input->producer;
      std::replace(previous->out.begin(), previous->out.end(), input, output);
      originals.erase(input->name);
      versions.erase(input->name);
    }

    removing.insert(unit);
    delete unit;
  }

  auto predicate = [&removing](Unit *unit) { return removing.count(unit) > 0; };
  units.erase(std::remove_if(units.begin(), units.end(), predicate),
      units.end());

  // Create units for the constants, sources and sinks.
  for (const auto &[_, wire] : originals) {
    if (wire->isConst && !wire->producer && wire->consumer) {
      getUnit("const", {}, { wire });
    } else if (wire->isInput && !wire->producer && wire->consumer) {
      auto *source = getUnit("source", {}, { wire });
      inputNamesToSources.insert({wire->name, source});
    } else if (wire->isOutput && wire->producer && !wire->consumer) {
      auto *sink = getUnit("sink", { wire }, {});
      outputNamesToSinks.insert({wire->name, sink});
    }
  }
  isTransformed = true;
}

//===----------------------------------------------------------------------===//
// Builder
//===----------------------------------------------------------------------===//

std::string Builder::getSourceName(const Unit *source) {
  return source->opcode + "_" + source->out.front()->name;
}

std::string Builder::getSinkName(const Unit *sink) {
  return sink->opcode + "_" + sink->in.front()->name;
}

std::shared_ptr<Model> Builder::create(const std::string &modelName) {
  auto *model = new Model(modelName);

  for (auto *kernel : kernels) {
    kernel->transform();
    model->addGraph(getGraph(kernel, model));
  }

  return std::shared_ptr<Model>(model);
}

std::shared_ptr<Model> Builder::create(const std::string &modelName,
                                       const std::string &kernelName) {
  auto *model = new Model(modelName);

  auto i = std::find_if(kernels.begin(), kernels.end(),
    [&kernelName](Kernel *kernel) { return kernel->name == kernelName; });
  assert(i != kernels.end() && "Kernel does not exist!\n");

  auto *kernel = *i;
  kernel->transform();
  model->addGraph(getGraph(kernel, model));

  return std::shared_ptr<Model>(model);
}

void Builder::startKernel(const std::string &name) {
  auto *kernel = new Kernel(name);

  kernel->originals.insert(common.originals.begin(), common.originals.end());
  kernel->versions.insert(common.versions.begin(), common.versions.end());

  kernels.push_back(kernel);
}

void Builder::declareWire(const ::dfc::wire *wire) {
  auto *kernel = getKernel();
  kernel->getWire(wire, Kernel::CREATE_ORIGINAL);
}

void Builder::connectWires(const ::dfc::wire *in,
                           const ::dfc::wire *out) {
  auto *kernel = getKernel();

  auto *source = kernel->getWire(in,  Kernel::ACCESS_VERSION);
  auto *target = kernel->getWire(out, Kernel::CREATE_VERSION);

  kernel->connect(source, target);
}

void Builder::connectToInstanceInput(const std::string &instanceName,
                                     const ::dfc::wire *wire,
                                     const std::string &inputName) {
  auto *kernel = getKernel();

  auto *source = kernel->getWire(wire, Kernel::ACCESS_VERSION);
  auto *target = kernel->getWire(wire, Kernel::CREATE_VERSION);

  kernel->connect(source, target);

  kernel->instanceUnits[instanceName]->addInput(target);

  getKernel()->instanceUnits[instanceName]->instance->addBindingToInput(target,
      inputName);
}

void Builder::connectToInstanceOutput(const std::string &instanceName,
                                      const ::dfc::wire *wire,
                                      const std::string &outputName) {
  auto *kernel = getKernel();
  auto *instanceKernel = getKernel(
      kernel->instanceUnits[instanceName]->instance->kernelName);

  auto *output = kernel->getWire(wire, Kernel::ACCESS_ORIGINAL);

  kernel->instanceUnits[instanceName]->addOutput(output);

  getKernel()->instanceUnits[instanceName]->instance->addBindingToOutput(output,
      instanceKernel->versions[outputName]->name);
}

void Builder::connectWires(const std::string &opcode,
                           const std::vector<const ::dfc::wire*> &in,
                           const std::vector<const ::dfc::wire*> &out) {
  auto *kernel = getKernel();

  // Create new inputs and connect them w/ the old ones.
  std::vector<Wire*> sources;
  std::vector<Wire*> targets;

  for (const auto *wire : in) {
    auto *source = kernel->getWire(wire, Kernel::ACCESS_VERSION);
    sources.push_back(source);
  }

  for (const auto *wire : in) {
    auto *target = kernel->getWire(wire, Kernel::CREATE_VERSION);
    targets.push_back(target);
  }

  for (std::size_t i = 0; i < in.size(); i++) {
    kernel->connect(sources[i], targets[i]);
  }

  // Compose outputs.
  std::vector<Wire*> outputs;
  for (const auto *wire : out) {
    auto *output = kernel->getWire(wire, Kernel::ACCESS_ORIGINAL);
    outputs.push_back(output);
  }

  // Create a unit w/ the newly created inputs.
  kernel->getUnit(opcode, targets, outputs);
}

Port *Builder::getPort(const Wire *wire, const unsigned latency) {
  return new Port(wire->name,                                       // Name
                  wire->type,                                       // Type
                  1.0,                                              // Flow
                  latency,                                          // Latency
                  wire->isConst,                                    // Constant
                  wire->value);                                     // Value 
}

Chan *Builder::getChan(const Wire *wire, Graph *graph) {
  auto *chan = graph->findChan(wire->name);

  if (!chan) {
    chan = new Chan(wire->name, wire->type.name, *graph);
    graph->addChan(chan);
  }

  return chan; 
}

NodeType *Builder::getNodetype(const Unit *unit,
                               Model *model) {
  auto *nodetype = new NodeType(unit->getFullName(), *model);
  for (auto *wire : unit->in) {
    nodetype->addInput(getPort(wire, 0));
  }
  for (auto *wire : unit->out) {
    nodetype->addOutput(getPort(wire, 1));
  }

  return nodetype;
}

Node *Builder::getNode(const Kernel *kernel,
                       const Unit *unit,
                       Graph *graph,
                       Model *model) {
  static unsigned id = 0;
  Signature signature = unit->getSignature();
  auto *nodetype = model->findNodetype(signature);
  if (nodetype == nullptr) {
    nodetype = getNodetype(unit, model);
    model->addNodetype(signature, nodetype);
  }
  std::string nodeName;
  if (unit->opcode == "source") {
    nodeName = getSourceName(unit);
  } else if (unit->opcode == "sink") {
    nodeName = getSinkName(unit);
  } else {
    nodeName = unit->opcode + std::to_string(id);
  }
  id++;
  Graph *instanceGraph = nullptr;
  if (unit->opcode == "INSTANCE") {
    // Check whether the instance is fully connected.
    if (!unit->instance->isFullyConnected()) {
      std::cout << "Instance " << unit->instance->instanceName << " " 
                << "is not fully connected!" << std::endl;
    }

    Kernel *instanceKernel = Builder::get().getKernel(
        unit->instance->kernelName);
    instanceGraph = model->findGraph(instanceKernel->name);
    if (instanceGraph == nullptr) {
      instanceGraph = getGraph(instanceKernel, model);
      model->addGraph(instanceGraph);
    }
    instanceGraph = model->findGraph(instanceKernel->name);
  }
  auto *node = new Node(nodeName, *nodetype, *graph, instanceGraph);
  for (auto *wire : unit->in) {
    auto *input = getChan(wire, graph);
    if (instanceGraph != nullptr) {
      Kernel *instanceKernel = Builder::get().getKernel(
          unit->instance->kernelName);
      Unit *sourceUnit = instanceKernel->inputNamesToSources[
          unit->instance->bindingsToInputs[wire]];
      Node *sourceNode = instanceGraph->findNode(getSourceName(sourceUnit));
      auto *con = new Con(input->name, input->type, "IN");
      con->source = { graph, input };
      con->target = { instanceGraph, sourceNode->outputs.back() };
      node->addCon(con);
    }
    input->target = { node, node->type.inputs[node->inputs.size()] };
    node->addInput(input);
  }
  for (auto *wire : unit->out) {
    auto *output = getChan(wire, graph);
    if (instanceGraph != nullptr) {
      Kernel *instanceKernel = Builder::get().getKernel(
          unit->instance->kernelName);
      Unit *sinkUnit = instanceKernel->outputNamesToSinks[
          unit->instance->bindingsToOutputs[wire]];
      Node *sinkNode = instanceGraph->findNode(getSinkName(sinkUnit));
      auto *con = new Con(output->name, output->type, "OUT");
      con->source = { instanceGraph, sinkNode->inputs.back() };
      con->target = { graph, output };
      node->addCon(con);
    }
    output->source = { node, node->type.outputs[node->outputs.size()] };
    node->addOutput(output);
  }
  return node;
}

Graph *Builder::getGraph(const Kernel *kernel, Model *model) {
  auto *graph = new Graph(kernel->name, *model);

  for (auto *unit : kernel->units) {
    graph->addNode(getNode(kernel, unit, graph, model));
  }

  return graph;
}

} // namespace eda::hls::parser::dfc