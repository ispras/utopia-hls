//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <unordered_map>

#include "hls/model/model.h"

namespace eda::hls::model {

void Graph::instantiate(
    const Graph &graph,
    const std::string &name,
    const std::map<std::string, std::map<std::string, Chan *>> &inputs,
    const std::map<std::string, std::map<std::string, Chan *>> &outputs) {

  // Maps original channel names to the created channel instances.
  std::unordered_map<std::string, Chan*> chans;

  // Clone the channels (except for the source outputs and the sink inputs).
  for (const auto *chan: graph.chans) {
    if (chan->source.node->is_source() || chan->target.node->is_sink()) {
      continue;
    }

    Chan *copy = new Chan(name + "." + chan->name, chan->type, *this);

    chans.insert({ chan->name, copy });
    add_chan(copy);
  }

  // Clone the nodes (except for the sources and sinks).
  for (const auto *node: graph.nodes) {
    if (node->is_source() || node->is_sink()) {
      continue;
    }

    Node *copy = new Node(name + "." + node->name, node->type, *this);

    for (const auto *input: node->inputs) {
      Chan *chan;

      if (input->source.node->is_source()) {
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

      assert(!chan->target.is_linked());
      chan->target = { copy, copy->type.inputs[copy->inputs.size()] };
      copy->add_input(chan);
    } // for inputs.

    for (const auto *output: node->outputs) {
      Chan *chan;

      if (output->target.node->is_sink()) {
        // Connect w/ the external channel from the bindings.
        auto i = outputs.find(output->target.node->name);
        assert(i != outputs.end());
        auto j = i->second.find(output->target.port->name);
        assert(j != i->second.end());
        chan = j->second;
      } else {
        // Connect w/ the internal channel instance.
        auto i = chans.find(output->name);
        if (i == chans.end()) std::cerr << "CHANNEL NOT FOUND: " << *output << std::endl;
        assert(i != chans.end());
        chan = i->second;
      }

      assert(!chan->source.is_linked());
      chan->source = { copy, copy->type.outputs[copy->outputs.size()] };
      copy->add_output(chan);
    } // for outputs.

    add_node(copy);
  } // for nodes.
}

std::ostream& operator <<(std::ostream &out, const Port &port) {
  out << port.type << "<" << port.flow << ">" << " ";
  out << "#" << port.latency << " " << port.name;
  if (port.is_const) {
    out << "=" << port.value;
  }
  return out;
}

static std::ostream& operator <<(std::ostream &out, const std::vector<Port *> &ports) {
  bool comma = false;
  for (const Port *port: ports) {
    out << (comma ? ", " : "") << *port;
    comma = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const NodeType &nodetype) {
  out << "  nodetype " << nodetype.name;
  return out << "(" << nodetype.inputs << ") => (" << nodetype.outputs << ");";
}

std::ostream& operator <<(std::ostream &out, const Chan &chan) {
  return out << "chan " << chan.type << " " << chan.name << ";";
}

static std::ostream& operator <<(std::ostream &out, const std::vector<Chan *> &params) {
  bool comma = false;
  for (const Chan *chan: params) {
    out << (comma ? ", " : "") << chan->name;
    comma = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const Node &node) {
  out << "node " << node.type.name << " " << node.name;
  return out << "(" << node.inputs << ") => (" << node.outputs << ");";
}

std::ostream& operator <<(std::ostream &out, const Graph &graph) {
  out << "  graph " << graph.name << " {" << std::endl;

  for (const Chan *chan: graph.chans) {
    out << "    " << *chan << std::endl;
  }

  for (const Node *node: graph.nodes) {
    out << "    " << *node << std::endl;
  }

  return out << "  }";
}

std::ostream& operator <<(std::ostream &out, const Model &model) {
  out << "model " << model.name << "{" << std::endl;

  for (const NodeType *nodetype: model.nodetypes) {
    out << *nodetype << std::endl;
  }

  for (const Graph *graph: model.graphs) {
    out << *graph << std::endl;
  }

  return out << "}";
}

} // namespace eda::hls::model
