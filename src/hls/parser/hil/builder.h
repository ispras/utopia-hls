//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "HIL/Ops.h"
#include "HIL/Utils.h"
#include "hls/model/model.h"
#include "util/assert.h"
#include "util/singleton.h"

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::parser::hil {

/**
 * \brief Helps to construct the IR from source code.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Builder final : public Singleton<Builder> {
  friend class Singleton<Builder>;

public:
  std::shared_ptr<Model> create();

  void startModel(const std::string &name) {
    assert(currentModel == nullptr && "Model is inside another model");

    currentModel = new Model(name);
    nodetypes.clear();
    graphs.clear();
  }

  void endModel() {}

  void startNodetype(const std::string &name) {
    assert(currentModel != nullptr && "Nodetype is outside a model");
    assert(currentNodetype == nullptr && "Previous nodetype has not been ended");

    currentNodetype = new NodeType(name, *currentModel);
    nodetypes.insert({ name, currentNodetype });
    outputsStarted = false;
  }

  void endNodetype() {
    assert(currentNodetype != nullptr && "Nodetype has not been started");

    currentModel->addNodetype(currentNodetype);
    currentNodetype = nullptr;
  }

  void startOutputs() {
    outputsStarted = true;
  }

  void addPort(
      const std::string &name,
      const std::string &type,
      const std::string &flow,
      const std::string &latency,
      const std::string &value = "") {
    auto *port = new Port(name,
                          type,
                          std::stof(flow),
                          std::stoi(latency),
                          !value.empty(),
                          value.empty() ? -1u : std::stoi(value));

    if (outputsStarted) {
      currentNodetype->addOutput(port);
    } else {
      assert(!port->isConst && "Input cannot be a const");
      currentNodetype->addInput(port);
    }
  }

  void startGraph(const std::string &name) {
    assert(currentModel != nullptr && "Graph is outside a model");
    assert(currentGraph == nullptr && "Previous graph has not been finished");

    currentGraph = new Graph(name, *currentModel);
    graphs.insert({ name, currentGraph });

    chans.clear();
  }

  void endGraph() {
    assert(currentGraph != nullptr);

    currentModel->addGraph(currentGraph);
    currentGraph = nullptr;
  }

  void addChan(const std::string &type,
      const std::string &name,
      const mlir::hil::BindingAttr &nodeFrom,
      const mlir::hil::BindingAttr &nodeTo) {

    assert(currentGraph != nullptr && "Chan is outside a graph");

    auto *chan = new Chan(name, type, *currentGraph);
    const auto *srcNode = currentGraph->findNode(nodeFrom.getNodeName());
    const auto *dstNode = currentGraph->findNode(nodeTo.getNodeName());
    auto fPort = nodeFrom.getPort();
    const std::string fPortName = fPort.getName();
    auto tPort = nodeTo.getPort();
    const std::string tPortName = tPort.getName();

    // FIXME: deprecated Type constructor is used here
    const auto *srcPort = new Port(fPortName,
        Type::get(fPort.getTypeName()),
        *fPort.getFlow(),
        fPort.getLatency(),
        fPort.getIsConst(),
        fPort.getValue());
    const auto *dstPort = new Port(tPortName,
        Type::get(tPort.getTypeName()),
        *tPort.getFlow(),
        tPort.getLatency(),
        tPort.getIsConst(),
        tPort.getValue());

    chan->source.node = srcNode;
    chan->source.port = srcPort;
    chan->target.node = dstNode;
    chan->target.port = dstPort;

    chans.insert({ name, chan });
    currentGraph->addChan(chan);
  }

  void addChan(const std::string &type, const std::string &name) {

    assert(currentGraph != nullptr && "Chan is outside a graph");

    auto *chan = new Chan(name, type, *currentGraph);
    chans.insert({ name, chan });
    currentGraph->addChan(chan);
  }

  void startNode(const std::string &type, const std::string &name) {
    assert(currentGraph != nullptr && "Node is outside a graph");
    assert(currentNode == nullptr && "Previous node has not been ended");

    auto i = nodetypes.find(type);
    uassert(i != nodetypes.end(), "Nodetype is not found: " << type);

    currentNode = new Node(name, *(i->second), *currentGraph);
    outputsStarted = false;
  }

  void endNode() {
    assert(currentNode != nullptr);

    currentGraph->addNode(currentNode);
    currentNode = nullptr;
  }

  void addParam(const std::string &name) {
    auto i = chans.find(name);
    uassert(i != chans.end(), "Chan is not found: " << name);

    auto *chan = i->second;
    if (currentInstance == nullptr) {
      assert(currentNode != nullptr && "Param is outside a node");

      // Node parameter.
      if (outputsStarted) {
        uassert(!chan->source.isLinked(), "Chan is already linked: " << *chan);
        chan->source = {
          currentNode,
          currentNode->type.outputs[currentNode->outputs.size()]
        };
        currentNode->addOutput(chan);
      } else {
        uassert(!chan->target.isLinked(), "Chan is already linked: " << *chan);
        chan->target = {
          currentNode,
          currentNode->type.inputs[currentNode->inputs.size()]
        };
        currentNode->addInput(chan);
      }
    } else {
      // Instance parameter.
      assert(currentInstanceNode != nullptr && "Param is outside an instance node");

      if (outputsStarted) {
        // Outputs are sink inputs.
        const Port *port = currentInstanceNode->type.inputs[currentBindings.size()];
        currentBindings.insert({ port->name, chan });
      } else {
        // Inputs are source outputs.
        const Port *port = currentInstanceNode->type.outputs[currentBindings.size()];
        currentBindings.insert({ port->name, chan });
      }
    }
  }

  void startInstance(const std::string &type, const std::string &name) {
    assert(currentInstance == nullptr && "Previous instance has not been ended");

    auto i = graphs.find(type);
    uassert(i != graphs.end(), "Graph is not found: " << name);

    currentInstance = i->second;
    currentInstanceName = name;

    currentInputs.clear();
    currentOutputs.clear();

    outputsStarted = false;
  }

  void endInstance() {
    assert(currentGraph != nullptr && "Graph has not been setup");
    assert(currentInstance != nullptr && "Instance graph is null");

    currentGraph->instantiate(
      *currentInstance, currentInstanceName, currentInputs, currentOutputs);
    currentInstance = nullptr;
  }

  void startBinding(const std::string &name) {
    assert(currentInstanceNode == nullptr && "Bind is outside an instance node");
    assert(currentInstance != nullptr && "Instance node is outside an instance");

    currentInstanceNode = currentInstance->findNode(name);
    uassert(currentInstanceNode != nullptr, "Node is not found: " << name);

    currentBindings.clear();
  }

  void endBinding() {
    if (outputsStarted) {
      currentOutputs.insert({ currentInstanceNode->name, currentBindings });
    } else {
      currentInputs.insert({ currentInstanceNode->name, currentBindings });
    }
    currentInstanceNode = nullptr;
  }

private:
  Builder() {}

  Model *currentModel = nullptr;
  NodeType *currentNodetype = nullptr;
  Graph *currentGraph = nullptr;
  Node *currentNode = nullptr;
  bool outputsStarted = false;

  Graph *currentInstance = nullptr;
  std::string currentInstanceName;
  Node *currentInstanceNode = nullptr;
  std::map<std::string, Chan*> currentBindings;

  std::map<std::string, std::map<std::string, Chan*>> currentInputs;
  std::map<std::string, std::map<std::string, Chan*>> currentOutputs;

  std::unordered_map<std::string, NodeType*> nodetypes;
  std::unordered_map<std::string, Chan*> chans;
  std::unordered_map<std::string, Graph*> graphs;
};

} // namespace eda::hls::parser::hil
