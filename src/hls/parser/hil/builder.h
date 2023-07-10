//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
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
#include "utils/assert.h"
#include "utils/singleton.h"

using Binding = eda::hls::model::Binding;
using BindingGraph = eda::hls::model::BindingGraph;
using Chan = eda::hls::model::Chan;
using Con = eda::hls::model::Con;
using Graph = eda::hls::model::Graph;
using Model = eda::hls::model::Model;
using Node = eda::hls::model::Node;
using NodeType = eda::hls::model::NodeType;
using Port = eda::hls::model::Port;
using Signature = eda::hls::model::Signature;
template<typename T>
using Singleton = eda::utils::Singleton<T>;
using Value = eda::hls::model::Value;


namespace eda::hls::parser::hil {

/**
 * \brief Helps to construct the IR from source code.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Builder final : public Singleton<Builder> {
  friend class Singleton<Builder>;

public:
  std::shared_ptr<Model> create();

  // Temporal solution.
  Signature getNodeTypeSignature(const std::string &nodeTypeName,
      const std::vector<std::string> inputChanNames,
      const std::vector<std::string> outputChanNames) {
    std::vector<std::string> inputTypeNames;
    std::vector<std::string> outputTypeNames;
    for (const auto &inputChanName : inputChanNames) {
      auto *chan = chans.find(inputChanName)->second;
      inputTypeNames.push_back(chan->type);
    }
    for (const auto &outputChanName : outputChanNames) {
      auto *chan = chans.find(outputChanName)->second;
      outputTypeNames.push_back(chan->type);
    }
    return Signature(nodeTypeName, inputTypeNames, outputTypeNames);
  }

  void startModel(const std::string &name) {
    uassert(currentModel == nullptr,
            "Model is inside another model!\n");

    currentModel = new Model(name);
    nodetypes.clear();
    graphs.clear();
  }

  void endModel() {}

  void startNodetype(const std::string &name) {
    uassert(currentModel != nullptr, "Nodetype is outside a model!\n");
    uassert(currentNodetype == nullptr,
            "Previous nodetype has not been ended!\n");

    currentNodetype = new NodeType(name, *currentModel);
    outputsStarted = false;
  }

  void endNodetype() {
    uassert(currentNodetype != nullptr, "Nodetype has not been started!\n");
    Signature signature = currentNodetype->getSignature();
    nodetypes.insert({signature, currentNodetype});
    currentModel->addNodetype(signature, currentNodetype);
    currentNodetype = nullptr;
  }

  void startOutputs() {
    outputsStarted = true;
  }

  void addPort(const std::string &name,
               const std::string &type,
               const std::string &flow,
               const std::string &latency,
               const std::string &value = "") {
    auto *port = new Port(name,
                          type,
                          std::stof(flow),
                          std::stoi(latency),
                          !value.empty(),
                          value.empty() ? Value(-1) : Value(value));
    if (outputsStarted) {
      currentNodetype->addOutput(port);
    } else {
      currentNodetype->addInput(port);
    }
  }

  void startGraph(const std::string &name) {
    uassert(currentModel != nullptr, "Graph is outside a model!\n");
    uassert(currentGraph == nullptr, "Previous graph has not been finished!\n");

    currentGraph = new Graph(name, *currentModel);
    graphs.insert({ name, currentGraph });

    chans.clear();
    cons.clear();
  }

  void endGraph() {
    uassert(currentGraph != nullptr, "Graph has not been started!\n");
    currentModel->addGraph(currentGraph);
    currentGraph = nullptr;
  }

  void addChan(const std::string &type,
               const std::string &name,
               const mlir::hil::BindingAttr &nodeFrom,
               const mlir::hil::BindingAttr &nodeTo) {

    uassert(currentGraph != nullptr, "Chan is outside a graph!\n");

    auto *chan = new Chan(name, type, *currentGraph);
    auto *graph = graphs.find(chan->graph.name)->second;
    /// TODO: Why was it needed?
    // const auto *srcNode = graph->findNode(nodeFrom.getNodeName().str());
    // const auto *dstNode = graph->findNode(nodeTo.getNodeName().str());
    auto fPort = nodeFrom.getPort();
    const std::string fPortName = fPort.getName();
    auto tPort = nodeTo.getPort();
    const std::string tPortName = tPort.getName();

    // FIXME: deprecated Type constructor is used here
    const auto *srcPort = new Port(fPortName,
                                   fPort.getTypeName(),
                                   fPort.getFlow(),
                                   fPort.getLatency(),
                                   fPort.getIsConst(),
                                   fPort.getValue());
    const auto *dstPort = new Port(tPortName,
                                   tPort.getTypeName(),
                                   tPort.getFlow(),
                                   tPort.getLatency(),
                                   tPort.getIsConst(),
                                   tPort.getValue());
    /// Why was it needed?
    // chan->source.node = srcNode;
    // chan->target.node = dstNode;
    chan->source.port = srcPort;
    chan->target.port = dstPort;

    chans.insert({ name, chan });
    graph->addChan(chan);
  }

  void addChan(const std::string &type, const std::string &name) {

    uassert(currentGraph != nullptr, "Chan is outside a graph!\n");

    auto *chan = new Chan(name, type, *currentGraph);
    chans.insert({ name, chan });
    currentGraph->addChan(chan);
  }

  void startInst(const std::string &name) {
    currentInstanceName = name;
  }

  void endInst() {
    // Do nothing.
  }

  void addCon(const std::string &name,
              const std::string &type,
              const std::string &dir,
              const mlir::hil::BindingGraphAttr &nodeFrom,
              const mlir::hil::BindingGraphAttr &nodeTo) {
    uassert(currentGraph != nullptr, "Con is outside a graph!\n");

    auto *con = new Con(name, type, dir);

    const auto *srcGraph = graphs.find(nodeFrom.getGraphName().str())->second;
    const auto *dstGraph = graphs.find(nodeTo.getGraphName().str())->second;
    auto *srcChan = srcGraph->findChan(nodeFrom.getChanName().str());
    auto *dstChan = dstGraph->findChan(nodeTo.getChanName().str());

    con->source = { srcGraph, srcChan };
    con->target = { dstGraph, dstChan };

    auto iterator = cons.find(currentInstanceName);
    if (iterator != cons.end()) {
      iterator->second.push_back(con);
    } else {
      std::vector<Con*> consToInstance;
      consToInstance.push_back(con);
      cons.insert({ currentInstanceName, consToInstance });
    }

    currentGraph->addCon(currentInstanceName, con);
  }

  void addCon(const std::string &name,
              const std::string &type,
              const std::string &dir) {
    uassert(currentGraph != nullptr, "Con is outside a graph!\n");

    auto *con = new Con(name, type, dir);

    auto iterator = cons.find(currentInstanceName);
    if (iterator != cons.end()) {
      iterator->second.push_back(con);
    } else {
      std::vector<Con*> consToInstance;
      consToInstance.push_back(con);
      cons.insert({ currentInstanceName, consToInstance });
    }
  }

  /// FIXME: deprecated
  void startNode(const std::string &type, const std::string &name) {
    uassert(currentGraph != nullptr, "Node is outside a graph!\n");
    uassert(currentNode == nullptr, "Previous node has not been ended!\n");

    auto i = nodetypesDeprecated.find(type);
    uassert(i != nodetypesDeprecated.end(), "Nodetype is not found: "
            << type << "!\n");

    currentNode = new Node(name, *(i->second), *currentGraph);
    outputsStarted = false;
  }

  void startNode(const Signature &typeSignature, const std::string &name) {
    uassert(currentGraph != nullptr, "Node is outside a graph!\n");
    uassert(currentNode == nullptr, "Previous node has not been ended!\n");

    auto i = nodetypes.find(typeSignature);
    uassert(i != nodetypes.end(), "Nodetype is not found: "
            << typeSignature.name << "!\n");

    currentNode = new Node(name, *(i->second), *currentGraph);
    outputsStarted = false;
  }

  void endNode() {
    uassert(currentNode != nullptr, "Node has not been started!\n");

    currentGraph->addNode(currentNode);
    currentNode = nullptr;
  }

  void addParam(const std::string &name) {
    auto i = chans.find(name);
    uassert(i != chans.end(), "Chan is not found: " << name << "!\n");

    auto *chan = i->second;
    if (currentInstance == nullptr) {
      uassert(currentNode != nullptr, "Param is outside a node!\n");

      // Node parameter.
      if (outputsStarted) {
        uassert(!chan->source.isLinked(), "Chan is already linked: " << *chan
                << "!\n");
        chan->source = {
          currentNode,
          currentNode->type.outputs[currentNode->outputs.size()]
        };
        currentNode->addOutput(chan);
      } else {
        uassert(!chan->target.isLinked(), "Chan is already linked: " << *chan
                << "!\n");
        chan->target = {
          currentNode,
          currentNode->type.inputs[currentNode->inputs.size()]
        };
        currentNode->addInput(chan);
      }
    } else {
      // Instance parameter.
      uassert(currentInstanceNode != nullptr,
              "Param is outside an instance node!\n");

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
    uassert(currentInstance == nullptr,
            "Previous instance has not been ended!\n");

    auto i = graphs.find(type);
    uassert(i != graphs.end(), "Graph is not found: " << name << "!\n");

    currentInstance = i->second;
    currentInstanceName = name;

    currentInputs.clear();
    currentOutputs.clear();

    outputsStarted = false;
  }

  void endInstance() {
    uassert(currentGraph != nullptr, "Graph has not been setup!\n");
    uassert(currentInstance != nullptr, "Instance graph is null!\n");

    currentGraph->instantiate(
      *currentInstance, currentInstanceName, currentInputs, currentOutputs);
    currentInstance = nullptr;
  }

  void startBinding(const std::string &name) {
    uassert(currentInstanceNode == nullptr,
            "Binding is outside an instance node!\n");
    uassert(currentInstance != nullptr,
            "Instance node is outside an instance!\n");

    currentInstanceNode = currentInstance->findNode(name);
    uassert(currentInstanceNode != nullptr, "Node is not found: " << name
            << "!\n");

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
  Node *currentInstanceNode = nullptr;
  std::map<std::string, Chan*> currentBindings;

  std::map<std::string, std::map<std::string, Chan*>> currentInputs;
  std::map<std::string, std::map<std::string, Chan*>> currentOutputs;

  std::unordered_map<Signature, NodeType*> nodetypes;
  std::unordered_map<std::string, NodeType*> nodetypesDeprecated;
  std::unordered_map<std::string, Chan*> chans;
  std::unordered_map<std::string, std::vector<Con*>> cons;
  std::string currentInstanceName;
  std::unordered_map<std::string, Graph*> graphs;
};

} // namespace eda::hls::parser::hil