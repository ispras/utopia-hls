//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <unordered_map>

#include "hls/model/model.h"

#include "hls/library/internal/verilog/delay.h"
#include "hls/mapper/mapper.h"
#include "hls/model/transform.h"

namespace eda::hls::model {
//===----------------------------------------------------------------------===//
// Signature
//===----------------------------------------------------------------------===//

bool Signature::operator==(const Signature &signature) const {
  if (name != signature.name) {
    return false;
  }

  if (inputTypeNames.size() != signature.inputTypeNames.size() ||
      outputTypeNames.size() != signature.outputTypeNames.size()) {
    return false;
  }

  for (size_t i = 0; i < inputTypeNames.size(); i++) {
    if (inputTypeNames[i] != signature.inputTypeNames[i]) {
      return false;
    }
  }
  for (size_t i = 0; i < outputTypeNames.size(); i++) {
    if (outputTypeNames[i] != signature.outputTypeNames[i]) {
      return false;
    }
  }

  return true;
}

Signature::Signature(const NodeType &nodeType) {
  name = nodeType.name;
  for (const auto *input : nodeType.inputs) {
    inputTypeNames.push_back(input->type.name);
  }
  for (const auto *output : nodeType.outputs) {
    outputTypeNames.push_back(output->type.name);
  }
}

Signature::Signature(const std::string &name, 
                     const std::vector<std::string> &inputTypeNames,
                     const std::vector<std::string> &outputTypeNames) {
  this->name = name;
  for (const auto &inputTypeName : inputTypeNames) {
    this->inputTypeNames.push_back(inputTypeName);
  }
  for (const auto &outputTypeName : outputTypeNames) {
    this->outputTypeNames.push_back(outputTypeName);
  }
}

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

const Type& Type::get(const std::string &name) {
  static std::unordered_map<std::string, std::unique_ptr<Type>> types;

  auto i = types.find(name);
  if (i != types.end())
    return *i->second;

  auto *type = new CustomType(name);
  types.insert({ name, std::unique_ptr<Type>(type) });

  return *type;
}

//===----------------------------------------------------------------------===//
// Graph
//===----------------------------------------------------------------------===//

void Graph::instantiate(
    const Graph &graph,
    const std::string &name,
    const std::map<std::string, std::map<std::string, Chan*>> &inputs,
    const std::map<std::string, std::map<std::string, Chan*>> &outputs) {

  // Maps original channel names to the created channel instances.
  std::unordered_map<std::string, Chan*> chans;

  // Clone the channels (except for the source outputs and the sink inputs).
  for (const auto *chan: graph.chans) {
    if (chan->source.node->isSource() || chan->target.node->isSink())
      continue;

    Chan *copy = new Chan(name + "_" + chan->name, chan->type, *this);

    chans.insert({ chan->name, copy });
    addChan(copy);
  }

  // Clone the nodes (except for the sources and sinks).
  for (const auto *node: graph.nodes) {
    if (node->isSource() || node->isSink())
      continue;

    Node *copy = new Node(name + "_" + node->name, node->type, *this);

    for (const auto *input: node->inputs) {
      Chan *chan;

      if (input->source.node->isSource()) {
        // Connect w/ the external channel from the bindings.
        auto i = inputs.find(input->source.node->name);
        assert(i != inputs.end());
        auto j = i->second.find(input->source.port->name);
        assert(j != i->second.end());
        chan = j->second;
      } else {
        // Connect w/ the internal channel instance.
        auto i = chans.find(input->name);
        assert(i != chans.end());
        chan = i->second;
      }

      assert(!chan->target.isLinked());
      chan->target = { copy, copy->type.inputs[copy->inputs.size()] };
      copy->addInput(chan);
    } // For inputs.

    for (const auto *output: node->outputs) {
      Chan *chan;

      if (output->target.node->isSink()) {
        // Connect w/ the external channel from the bindings.
        auto i = outputs.find(output->target.node->name);
        assert(i != outputs.end());
        auto j = i->second.find(output->target.port->name);
        assert(j != i->second.end());
        chan = j->second;
      } else {
        // Connect w/ the internal channel instance.
        auto i = chans.find(output->name);
        assert(i != chans.end());
        chan = i->second;
      }

      assert(!chan->source.isLinked());
      chan->source = { copy, copy->type.outputs[copy->outputs.size()] };
      copy->addOutput(chan);
    } // For outputs.

    addNode(copy);
  } // For nodes.
}

//===----------------------------------------------------------------------===//
// Model
//===----------------------------------------------------------------------===//

void Model::save() {
  for (auto *transform : transforms)
    delete transform;

  transforms.clear();
}

void Model::undo() {
  for (auto *transform : transforms)
    transform->undo();

  save();
}

void Model::insertDelay(Chan &chan, unsigned latency) {
  auto *transform = new InsertDelay(*this, chan, latency);
  transforms.push_back(transform);

  transform->apply();

  // Get the inserted delay node.
  auto *delay = transform->newNodes.back();
  assert(delay && "Inserted delay node is not found");

  // Map the node to the corresponding meta-element.
  mapper::Mapper::get().map(*delay, library::Library::get());
  assert(delay->map && "Node is unmapped");

  // Set the latency parameter.
  Parameters params(delay->map->params);
  params.setValue(library::internal::verilog::Delay::depth, latency);

  // Apply the parameters to the node.
  mapper::Mapper::get().apply(*delay, params);
}

//===----------------------------------------------------------------------===//
// Output
//===----------------------------------------------------------------------===//

std::ostream& operator<<(std::ostream &out, const Type &type) {
  return out << type.name;
}

std::ostream& operator<<(std::ostream &out, const Port &port) {
  out << port.type << "<" << port.flow << ">" << " ";
  out << "#" << port.latency << " " << port.name;
  if (port.isConst)
    out << "=" << port.value.getIntValue();
  return out;
}

static std::ostream& operator<<(std::ostream &out,
                                const std::vector<Port*> &ports) {
  bool comma = false;
  for (const auto *port: ports) {
    out << (comma ? ", " : "") << *port;
    comma = true;
  }

  return out;
}

std::ostream& operator<<(std::ostream &out, const NodeType &nodetype) {
  out << "  nodetype " << nodetype.name;
  return out << "(" << nodetype.inputs << ") => (" << nodetype.outputs << ");";
}

std::ostream& operator<<(std::ostream &out, const Chan &chan) {
    return out << "chan " << chan.type << " " << chan.name << " "
             << "(graph: " << chan.graph.name << " node: "
             << chan.source.node->name << " port: "
             << chan.source.port->name << ")"
             << " => " << "(graph: " << chan.graph.name << " node: "
             << chan.target.node->name << " port: "
             << chan.target.port->name << ")";
}

static std::ostream& operator<<(std::ostream &out,
                                const std::vector<Chan*> &chans) {
  bool comma = false;
  for (const auto *chan: chans) {
    out << (comma ? ", " : "") << chan->name;
    comma = true;
  }

  return out;
}

std::ostream& operator<<(std::ostream &out, const Con &con) {
  return out << "con " << con.type << " " << con.name << " " << con.dir << " {" 
             << std::endl << "        "
             << "graph " << con.source.graph->name << " " << *con.source.chan 
             << std::endl << "        "
             << "graph " << con.target.graph->name << " " << *con.target.chan 
             << std::endl << "      " << "}";
}

std::ostream& operator<<(std::ostream &out, const Node &node) {
  out << "node " << node.type.name << " " << node.name;
  return out << "(" << node.inputs << ") => (" << node.outputs << ");";
}

std::ostream& operator<<(std::ostream &out, const Graph &graph) {
  out << "  graph " << graph.name << " {" << std::endl;

  for (const auto *chan: graph.chans)
    out << "    " << *chan << std::endl;

  for (const auto *node: graph.nodes)
    out << "    " << *node << std::endl;

  for (const auto &pair: graph.cons) {
    out << "    " << pair.first << ": {" << std::endl;

    for (const auto *con: pair.second) {
      out << "      " << *con << std::endl; 
    }
    out << "    }" << std::endl;
  }


  return out << "  }";
}

std::ostream& operator<<(std::ostream &out, const Model &model) {
  out << "model " << model.name << "{" << std::endl;
  
  for (auto nodeTypeIterator = model.nodetypes.begin();
       nodeTypeIterator != model.nodetypes.end();
       nodeTypeIterator++) {
    out << *nodeTypeIterator->second << std::endl;
  }

  for (const auto *graph: model.graphs)
    out << *graph << std::endl;

  return out << "}" << std::endl;
}

} // namespace eda::hls::model