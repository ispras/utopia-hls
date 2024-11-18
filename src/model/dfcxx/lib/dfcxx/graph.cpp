//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/graph.h"

#include <algorithm>
#include <stdexcept>

namespace dfcxx {

const std::unordered_set<Node> &Graph::getNodes() const {
  return nodes;
}

const std::unordered_set<Node> &Graph::getStartNodes() const {
  return startNodes;
}

const std::unordered_map<Node, std::vector<Channel>> &Graph::getInputs() const {
  return inputs;
}

const std::unordered_map<Node, std::vector<Channel>> &Graph::getOutputs() const {
  return outputs;
}

const std::unordered_map<Node, Channel> &Graph::getConnections() const {
  return connections;
}

Node Graph::findNode(DFVariableImpl *var) {
  return *std::find_if(nodes.begin(), nodes.end(),
                       [&](const Node &node) { return node.var == var; });
}

void Graph::addNode(DFVariableImpl *var, OpType type, NodeData data) {
  auto node = nodes.emplace(var, type, data);
  if (type == IN || type == CONST) {
    startNodes.emplace(var, type, data);
  }

  auto name = node.first->var->getName();
  if (!name.empty()) {
    nameMap[name] = *(node.first);
  }

  // The following lines create empty channel vectors
  // for new nodes. This allows to use .at() on unconnected
  // nodes without getting an exception.
  (void)inputs[*(node.first)];
  (void)outputs[*(node.first)];
}

void Graph::addChannel(DFVariableImpl *source, DFVariableImpl *target,
                       unsigned opInd, bool connect) {
  Node foundSource = findNode(source);
  Node foundTarget = findNode(target);
  Channel newChannel(foundSource, foundTarget, opInd);
  outputs[foundSource].push_back(newChannel);
  inputs[foundTarget].push_back(newChannel);
  if (connect) {
    connections.insert(std::make_pair(foundTarget, newChannel));
    connections.at(foundTarget) = newChannel;
  }
}

void Graph::transferFrom(Graph &&graph) {
  nodes.merge(std::move(graph.nodes));
  inputs.merge(std::move(graph.inputs));
  outputs.merge(std::move(graph.outputs));
  connections.merge(std::move(graph.connections));
}

Node Graph::getNodeByName(const std::string &name) {
  auto it = nameMap.find(name);
  if (it == nameMap.end()) {
    throw new std::invalid_argument("Non-existent node with name: " + name);
  }
  return it->second;
}

void Graph::resetNodeName(const std::string &name) {
  auto it = nameMap.find(name);
  if (it == nameMap.end()) {
    throw new std::invalid_argument("Non-existent node with name: " + name);
  }
  DFVariableImpl *ptr = it->second.var;
  nameMap.erase(it);
  ptr->resetName();
}

void Graph::deleteNode(Node node) {
  connections.erase(node);
  inputs.erase(node);
  outputs.erase(node);
  startNodes.erase(node);
  nodes.erase(node);
}

void Graph::rebindInput(Node source, Node input, Graph &graph) {
  for (auto &out: graph.outputs[input]) {
    for (auto &in: graph.inputs[out.target]) {
      if (in.source == input) {
        in.source = source;
        outputs[source].push_back(in);
        break;
      }
    }
  }
  
  for (auto &conn: graph.connections) {
    if (conn.second.source == input) {
      conn.second.source = source;
    }
  }
}

Node Graph::rebindOutput(Node output, Node target, Graph &graph) {
  auto &in = graph.inputs[output].front().source;
  auto &outs = graph.outputs[in];
  for (auto it = outs.begin(); it != outs.end(); ++it) {
    if (it->target == output) {
      if (target.type == OpType::NONE) {
        target = it->source;
        outs.erase(it);
      } else {
        it->target = target;
        inputs[target].clear();
        inputs[target].push_back(*it);
        connections[target] = *it;
      }
      break;
    }
  }
  return target;
}

} // namespace dfcxx
