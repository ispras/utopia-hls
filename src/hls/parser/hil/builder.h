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

#include "hls/model/model.h"
#include "util/assert.h"

using namespace eda::hls::model;

namespace eda::hls::parser::hil {

/**
 * \brief Helps to contruct the IR from source code.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Builder final {
public:
  static Builder& get() {
    if (_instance == nullptr)
      _instance = std::unique_ptr<Builder>(new Builder());
    return *_instance;
  }

  std::unique_ptr<Model> create();

  void start_model(const std::string &name) {
    assert(_model == nullptr && "Model is inside another model");

    _model = new Model(name);
    _nodetypes.clear();
    _graphs.clear();
  }

  void end_model() {}

  void start_nodetype(const std::string &name) {
    assert(_model != nullptr && "Nodetype is outside a model");
    assert(_nodetype == nullptr && "Previous nodetype has not been ended");

    _nodetype = new NodeType(name, *_model);
    _nodetypes.insert({ name, _nodetype });
    _outputs = false;
  }

  void end_nodetype() {
    assert(_nodetype != nullptr && "Nodetype has not been started");

    _model->addNodetype(_nodetype);
    _nodetype = nullptr; 
  }

  void start_outputs() {
    _outputs = true;
  }

  void add_port(
      const std::string &name,
      const std::string &type,
      const std::string &flow,
      const std::string &latency,
      const std::string &value = "") {
    Port *port = new Port(name,
                          type,
                          std::stof(flow),
                          std::stoi(latency),
                          !value.empty(),
                          value.empty() ? -1u : std::stoi(value));

    if (_outputs) {
      _nodetype->addOutput(port);
    } else {
      assert(!port->isConst && "Input cannot be a const");
      _nodetype->addInput(port);
    }
  }

  void start_graph(const std::string &name) {
    assert(_model != nullptr && "Graph is outside a model");
    assert(_graph == nullptr && "Previous graph has not been finished");

    _graph = new Graph(name, *_model);
    _graphs.insert({ name, _graph });

    _chans.clear();
  }

  void end_graph() {
    assert(_graph != nullptr);

    _model->addGraph(_graph);
    _graph = nullptr;
  }

  void add_chan(const std::string &type, const std::string &name) {
    assert(_graph != nullptr && "Chan is outside a graph");

    Chan *chan = new Chan(name, type, *_graph);
    _chans[name] = chan;
    _graph->addChan(chan);
  }

  void start_node(const std::string &type, const std::string &name) {
    assert(_graph != nullptr && "Node is outside a graph");
    assert(_node == nullptr && "Previous node has not been ended");

    auto i = _nodetypes.find(type);
    uassert(i != _nodetypes.end(), "Nodetype is not found: " << type);
 
    _node = new Node(name, *(i->second), *_graph);
    _outputs = false;
  }

  void end_node() {
    assert(_node != nullptr);

    _graph->addNode(_node);
    _node = nullptr;
  }

  void add_param(const std::string &name) {
    auto i = _chans.find(name);
    uassert(i != _chans.end(), "Chan is not found: " << name);

    Chan *chan = i->second;
    if (_inst_graph == nullptr) {
      assert(_node != nullptr && "Param is outside a node");

      // Node parameter.
      if (_outputs) {
        uassert(!chan->source.isLinked(), "Chan is already linked: " << *chan);
        chan->source = { _node, _node->type.outputs[_node->outputs.size()] };
        _node->addOutput(chan);
      } else {
        uassert(!chan->target.isLinked(), "Chan is already linked: " << *chan);
        chan->target = { _node, _node->type.inputs[_node->inputs.size()] };
        _node->addInput(chan);
      }
    } else {
      // Instance parameter.
      assert(_inst_node != nullptr && "Param is outside an instance node");

      if (_outputs) {
        // Outputs are sink inputs.
        const Port *port = _inst_node->type.inputs[_inst_node_binds.size()];
        _inst_node_binds.insert({ port->name, chan });
      } else {
        // Inputs are source outputs.
        const Port *port = _inst_node->type.outputs[_inst_node_binds.size()];
        _inst_node_binds.insert({ port->name, chan });
      }
    }
  }

  void start_inst(const std::string &type, const std::string &name) {
    assert(_inst_graph == nullptr && "Previous instance has not been ended");

    auto i = _graphs.find(type);
    uassert(i != _graphs.end(), "Graph is not found: " << name);

    _inst_graph = i->second;
    _inst_name = name;

    _inst_inputs.clear();
    _inst_outputs.clear();

    _outputs = false;
  }

  void end_inst() {
    assert(_graph != nullptr && "Graph has not been setup");
    assert(_inst_graph != nullptr && "Instance graph is null");

    _graph->instantiate(*_inst_graph, _inst_name, _inst_inputs, _inst_outputs);
    _inst_graph = nullptr;
  }

  void start_bind(const std::string &name) {
    assert(_inst_node == nullptr && "Bind is outside an instance node");
    assert(_inst_graph != nullptr && "Instance node is outside an instance");

    auto i = std::find_if(_inst_graph->nodes.begin(),
                          _inst_graph->nodes.end(),
                          [&name](Node *node) { return node->name == name; });
    uassert(i != _inst_graph->nodes.end(), "Node is not found: " << name);

    _inst_node = *i;
    _inst_node_binds.clear();
  }

  void end_bind() {
    if (_outputs) {
      _inst_outputs.insert({ _inst_node->name, _inst_node_binds });
    } else {
      _inst_inputs.insert({ _inst_node->name, _inst_node_binds });
    }
    _inst_node = nullptr;
  }

private:
  Builder() {}

  Model *_model = nullptr;
  NodeType *_nodetype = nullptr;
  Graph *_graph = nullptr;
  Node *_node = nullptr;
  bool _outputs = false;

  Graph *_inst_graph = nullptr;
  std::string _inst_name;
  std::map<std::string, std::map<std::string, Chan*>> _inst_inputs;
  std::map<std::string, std::map<std::string, Chan*>> _inst_outputs;
  Node *_inst_node;
  std::map<std::string, Chan*> _inst_node_binds;

  std::unordered_map<std::string, NodeType*> _nodetypes;
  std::unordered_map<std::string, Chan*> _chans;
  std::unordered_map<std::string, Graph*> _graphs;

  static std::unique_ptr<Builder> _instance;
};

} // namespace eda::hls::parser::hil
