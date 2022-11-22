//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/model/transform.h"

#include <iostream>

namespace eda::hls::model {

void InsertDelay::apply() {
  // deprecated
  /*std::string typeName = "delay_" + chan.type + "_" + std::to_string(latency);

  NodeType *nodetype = model.findNodetype(typeName);

  if (nodetype == nullptr) {
    nodetype = new NodeType(typeName, model);
    newTypes.push_back(nodetype);

    Port *input = new Port("in", chan.source.port->type, 1.0, 0, false, 0);
    newPorts.push_back(input);
    Port *output = new Port("out", chan.target.port->type, 1.0, latency, false,
        0);
    newPorts.push_back(output);

    nodetype->addInput(input);
    nodetype->addOutput(output);
    model.addNodetype(nodetype);
  }*/
  //Temporal solution.
  //const std::string nodeTypeName           = "delay";
  std::string nodeTypeName = "delay_" + chan.type + "_" + std::to_string(latency);
  std::vector<std::string> inputTypeNames  = {chan.type};
  std::vector<std::string> outputTypeNames = {chan.type};

  Signature signature(nodeTypeName, inputTypeNames, outputTypeNames);

  auto *nodeType = model.findNodetype(signature);

  if (nodeType == nullptr) {
    nodeType = new NodeType(nodeTypeName, model);
    newTypes.push_back(nodeType);

    Port *input = new Port("in", chan.source.port->type, 1.0, 0, false, 0);
    newPorts.push_back(input);
    Port *output = new Port("out",
                            chan.target.port->type,
                            1.0,
                            latency/*deprecated*/,
                            false,
                            0);
    newPorts.push_back(output);

    nodeType->addInput(input);
    nodeType->addOutput(output);
    model.addNodetype(signature, nodeType);
  }

  Graph &graph = chan.graph;

  std::string nodeName = nodeTypeName + "_" +
      std::to_string(graph.nodes.size());
  Node *node = new Node(nodeName, *nodeType, graph);
  newNodes.push_back(node);

  Chan *from = new Chan(nodeName + "_out", chan.type, graph, chan.latency);
  newChans.push_back(from);

  from->source = { node, nodeType->findOutput("out") };
  from->target = chan.target;
  chan.target = { node, nodeType->findInput("in") };
  chan.latency = latency;

  node->addInput(&chan);
  node->addOutput(from);

  auto &inputs = const_cast<std::vector<Chan*>&>(from->target.node->inputs);
  std::replace(inputs.begin(), inputs.end(), &chan, from);

  graph.addChan(from);
  graph.addNode(node);
}

void InsertDelay::undo() {
  const Node *node = chan.target.node;
  Chan *from = node->outputs[0];

  auto &inputs = const_cast<std::vector<Chan*>&>(from->target.node->inputs);
  std::replace(inputs.begin(), inputs.end(), from, &chan);

  chan.target = from->target;
  chan.latency = from->latency;

  Graph &graph = chan.graph;

  for (auto *type : newTypes) {
    Signature signature = type->getSignature();
    model.nodetypes.erase(signature);
    //deprecated
    /*auto i = std::find(model.nodetypes.begin(), model.nodetypes.end(), type);
    model.nodetypes.erase(i);*/
    delete type;
  }
  for (auto *chan : newChans) {
    auto i = std::find(graph.chans.begin(), graph.chans.end(), chan);
    graph.chans.erase(i);
    delete chan;
  }
  for (auto *node : newNodes) {
    auto i = std::find(graph.nodes.begin(), graph.nodes.end(), node);
    graph.nodes.erase(i);
    delete node;
  }
  for (auto *port : newPorts) {
    delete port;
  }
}

} // namespace eda::hls::model
