//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/graph.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace dfcxx {

Graph::~Graph() {
  for (Node *node: nodes) {
    delete node;
  }

  for (Channel *channel: channels) {
    delete channel;
  }
}

Node *Graph::findNode(const std::string &name) {
  auto it = nameMap.find(name);
  if (it == nameMap.end()) {
    throw new std::invalid_argument("Non-existent node with name: " + name);
  }
  return it->second;
}

Node *Graph::findNode(DFVariableImpl *var) {
  Node *bufNode = new Node(var);
  auto found = nodes.find(bufNode);
  delete bufNode;
  if (found != nodes.end()) {
    return *found;
  }
  return nullptr;
}

Node *Graph::findNodeIf(NodePtrPred pred) {
  auto found = std::find_if(nodes.begin(), nodes.end(), pred);
  return (found != nodes.end()) ? *found : nullptr;
}

Node *Graph::findStartNode(DFVariableImpl *var) {
  Node *bufNode = new Node(var);
  auto found = startNodes.find(bufNode);
  delete bufNode;
  if (found != startNodes.end()) {
    return *found;
  }
  return nullptr;
}

Node *Graph::findStartNodeIf(NodePtrPred pred) {
  auto found = std::find_if(startNodes.begin(), startNodes.end(), pred);
  return (found != startNodes.end()) ? *found : nullptr;
}

std::pair<Node *, bool> Graph::addNode(DFVariableImpl *var,
                                       OpType type,
                                       NodeData data) {
  Node *newNode = new Node(var, type, data);
  auto insertRes = nodes.insert(newNode);

  if (!insertRes.second) {
    delete newNode;
    return std::make_pair(*(insertRes.first), insertRes.second);
  }

  if (type == IN || type == CONST) {
    startNodes.insert(newNode);
  }

  auto name = (*(insertRes.first))->var->getName();
  if (!name.empty()) {
    nameMap[name] = *(insertRes.first);
  }

  // The following lines create empty channel vectors
  // for new nodes. This allows to use .at() on unconnected
  // nodes without getting an exception.
  (void)inputs[*(insertRes.first)];
  (void)outputs[*(insertRes.first)];
  return std::make_pair(*(insertRes.first), insertRes.second);
}

Channel *Graph::addChannel(Node *source, Node *target,
                           unsigned opInd, bool connect) {
  assert(source && target);
  Channel *newChannel = new Channel(source, target, opInd);

  outputs[source].push_back(newChannel);
  inputs[target].push_back(newChannel);
  if (connect) {
    connections[target] = newChannel;
  }
  return newChannel;
}

Channel *Graph::addChannel(DFVariableImpl *source, DFVariableImpl *target,
                           unsigned opInd, bool connect) {
  Node *foundSource = findNode(source);
  Node *foundTarget = findNode(target);
  return addChannel(foundSource, foundTarget, opInd, connect);
}

void Graph::transferFrom(Graph &&graph) {
  nodes.merge(std::move(graph.nodes));
  inputs.merge(std::move(graph.inputs));
  outputs.merge(std::move(graph.outputs));
  connections.merge(std::move(graph.connections));
}

void Graph::resetNodeName(const std::string &name) {
  auto it = nameMap.find(name);
  if (it == nameMap.end()) {
    throw new std::invalid_argument("Non-existent node with name: " + name);
  }
  nameMap.erase(it);
  it->second->var->resetName();
}

void Graph::deleteNode(Node *node) {
  connections.erase(node);
  inputs.erase(node);
  outputs.erase(node);
  startNodes.erase(node);
  nodes.erase(node);
}

void Graph::rebindInput(Node *source, Node *input, Graph &graph) {
  auto &conns = graph.connections;
  for (auto out: graph.outputs[input]) {
    for (auto in: graph.inputs[out->target]) {
      if (in->source == input && out == in) {
        in->source = source;
        outputs[source].push_back(in);
        break;
      }
    }
    auto it = conns.find(out->target);
    if (it != conns.end() && it->second->source == input) {
      it->second->source = source;
    }
  }
}

Node *Graph::rebindOutput(Node *output, Node *target, Graph &graph) {
  Node *inSrc = graph.inputs[output].front()->source;
  auto &outs = graph.outputs[inSrc];
  for (auto it = outs.begin(); it != outs.end(); ++it) {
    if ((*it)->target != output) { continue; }
    if (target->type == OpType::NONE) {
      outs.erase(it);
      for (auto out: outputs[target]) {
        for (auto in: inputs[out->target]) {
          if (in->source == target && out == in) {
            in->source = (*it)->source;
            outs.push_back(in);
          }
        }
        auto conIt = connections.find(out->target);
        if (conIt != connections.end() && conIt->second->source == target) {
          conIt->second->source = (*it)->source;
        }
      }
      target = (*it)->source;
    } else {
      (*it)->target = target;
      inputs[target].clear();
      inputs[target].push_back(*it);
      connections[target] = *it;
    }
    break;
  }
  return target;
}

} // namespace dfcxx
