//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "hls/model/model.h"

#define CHECK(expr, message) \
  do {\
    if (!(expr)) std::cerr << message << std::endl; \
    assert(expr); \
  } while(false)

using namespace eda::hls::model;

namespace eda::hls::parser::hil {

/**
 * \brief Helps to contruct the IR from source code.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Builder final {
public:
  static Builder& get() {
    if (_instance == nullptr) {
      _instance = std::unique_ptr<Builder>(new Builder());
    }
    return *_instance;
  }

  std::unique_ptr<Model> create();

  void start_model(const std::string &name) {
    CHECK(_model == nullptr, "Model is not null");

    _model = new Model(name);
    _nodetypes.clear();
  }

  void end_model() {}

  void start_nodetype(const std::string &name) {
    CHECK(_model != nullptr, "Model is null");
    CHECK(_nodetype == nullptr, "NodeType is not null");

    _nodetype = new NodeType(name, *_model);
    _nodetypes.insert({ name, _nodetype });
    _outputs = false;
  }

  void end_nodetype() {
    CHECK(_nodetype != nullptr, "NodeType is null");

    _model->add_nodetype(_nodetype);
    _nodetype = nullptr; 
  }

  void start_outputs() {
    _outputs = true;
  }

  void add_argument(
      const std::string &name,
      const std::string &type,
      const std::string &flow,
      const std::string &latency,
      const std::string &value = "") {
    Argument *argument = new Argument(name,
                                      type,
                                      std::stof(flow),
                                      std::stoi(latency),
                                      !value.empty(),
                                      value.empty() ? -1u : std::stoi(value));

    if (_outputs) {
      _nodetype->add_output(argument);
    } else {
      CHECK(!argument->is_const, "Input is const");
      _nodetype->add_input(argument);
    }
  }

  void start_graph(const std::string &name) {
    CHECK(_model != nullptr, "Model is null");
    CHECK(_graph == nullptr, "Graph is not null");

    _graph = new Graph(name, *_model);
    _chans.clear();
  }

  void end_graph() {
    CHECK(_graph != nullptr, "Graph is null");

    _model->add_graph(_graph);
    _graph = nullptr;
  }

  void add_chan(const std::string &type, const std::string &name) {
    CHECK(_graph != nullptr, "Graph is null");

    Chan *chan = new Chan(name, type, *_graph);
    _chans[name] = chan;
    _graph->add_chan(chan);
  }

  void start_node(const std::string &type, const std::string &name) {
    CHECK(_graph != nullptr, "Graph is null");
    CHECK(_node == nullptr, "Node is not null");

    const auto &i = _nodetypes.find(type);
    CHECK(i != _nodetypes.end(), "NodeType is not found: " << type);
 
    _node = new Node(name, *(i->second), *_graph);
    _outputs = false;
  }

  void end_node() {
    CHECK(_node != nullptr, "Node is null");

    _graph->add_node(_node);
    _node = nullptr;
  }

  void add_param(const std::string &name) {
    CHECK(_node != nullptr, "Node is null");

    const auto &i = _chans.find(name);
    CHECK(i != _chans.end(), "Chan is not found: " << name);

    Chan *chan = i->second;
    if (_outputs) {
      CHECK(!chan->source.is_linked(), "Chan is already linked: " << *chan);
      chan->source = { _node, _node->type.outputs[_node->outputs.size()] };
      _node->add_output(chan);
    } else {
      CHECK(!chan->target.is_linked(), "Chan is already linked: " << *chan);
      chan->target = { _node, _node->type.inputs[_node->inputs.size()] };
      _node->add_input(chan);
    }
  }

private:
  Builder() {}

  Model *_model = nullptr;
  NodeType *_nodetype = nullptr;
  Graph *_graph = nullptr;
  Node *_node = nullptr;
  bool _outputs = false;

  std::unordered_map<std::string, NodeType *> _nodetypes;
  std::unordered_map<std::string, Chan *> _chans;

  static std::unique_ptr<Builder> _instance;
};

} // namespace eda::hls::parser::hil
