/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "hls/model/model.h"

using namespace eda::hls::model;

namespace eda::hls::parser {

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

  void start_model() {
    assert(_model == nullptr);

    _model = new Model();
    _nodetypes.clear();
    _chans.clear();
  }

  void end_model() {}

  void start_nodetype(const std::string &name, const std::string &latency) {
    assert(_nodetype == nullptr);

    _nodetype = new NodeType(name, ::atoi(latency.c_str()));
    _nodetypes.insert({ name, _nodetype });
    _outputs = false;
  }

  void end_nodetype() {
    assert(_nodetype != nullptr);

    _model->add_nodetype(_nodetype);
    _nodetype = nullptr; 
  }

  void start_outputs() {
    _outputs = true;
  }

  void add_argument(const std::string &name, const std::string &type, const std::string &flow) {
    Argument *argument = new Argument(name, type, ::atof(flow.c_str()));

    if (_outputs) {
      _nodetype->add_output(argument);
    } else {
      _nodetype->add_input(argument);
    }
  }

  void start_graph(const std::string &name) {
    assert(_graph == nullptr);
    _graph = new Graph(name);
  }

  void end_graph() {
    assert(_graph != nullptr);

    _model->add_graph(_graph);
    _graph = nullptr;
  }

  void add_chan(const std::string &type, const std::string &name) {
    assert(_graph != nullptr);

    Chan *chan = new Chan(name, type);
    _chans[name] = chan;
    _graph->add_chan(chan);
  }

  void start_node(const std::string &type) {
    assert(_node == nullptr);

    const auto &i = _nodetypes.find(type);
    assert(i != _nodetypes.end());
 
    _node = new Node(*(i->second));
    _outputs = false;
  }

  void end_node() {
    assert(_node != nullptr);

    _graph->add_node(_node);
    _node = nullptr;
  }

  void add_param(const std::string &name) {
    assert(_node != nullptr);

    const auto &i = _chans.find(name);
    assert(i != _chans.end());

    Chan *chan = i->second;
    if (_outputs) {
      assert(chan->source == nullptr);
      chan->source = _node->type.outputs[_node->outputs.size()];
      _node->add_output(chan);
    } else {
      assert(chan->target == nullptr);
      chan->target = _node->type.inputs[_node->inputs.size()];
      _node->add_input(chan);
    }
  }

private:
  Builder() {}

  Model *_model = nullptr;
  Argument *_argument = nullptr;
  NodeType *_nodetype = nullptr;
  Graph *_graph = nullptr;
  Node *_node = nullptr;
  bool _outputs = false;

  std::unordered_map<std::string, NodeType *> _nodetypes;
  std::unordered_map<std::string, Chan *> _chans;

  static std::unique_ptr<Builder> _instance;
};

} // namespace eda::hls::parser
