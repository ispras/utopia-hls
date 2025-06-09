//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/graph.h"
#include "dfcxx/kernel_meta.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace dfcxx {

class Kernel {
  static inline KernelMeta* getTopMeta();
};

Graph::~Graph() {
  // By convention to delete a Channel object
  // each input of each node is deleted.
  for (Node *node: nodes) {
    for (Channel *channel: node->outputs) {
      delete channel;
    }
    delete node;
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

  return std::make_pair(*(insertRes.first), insertRes.second);
}

Channel *Graph::addChannel(Node *source, Node *target,
                           unsigned opInd, bool connect) {
  assert(source && target);
  Channel *newChannel = new Channel(source, target, opInd, connect);

  source->outputs.push_back(newChannel);
  target->inputs.push_back(newChannel);
  return newChannel;
}

Channel *Graph::addChannel(DFVariableImpl *source, DFVariableImpl *target,
                           unsigned opInd, bool connect) {
  // Constants are saved in top-level graph.
  Node *foundSource = (source->isConstant() ?
		      &(KernelMeta::top->graph) :
		      this)->findNode(source);
  Node *foundTarget = findNode(target);
  return addChannel(foundSource, foundTarget, opInd, connect);
}

void Graph::transferFrom(Graph &&graph) {
  nodes.merge(std::move(graph.nodes));
  for (Node *node: graph.startNodes) {
    if (node->type == OpType::CONST) {
      startNodes.insert(node);
    }
  }
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
  startNodes.erase(node);
  nodes.erase(node);
  delete node;
}

void Graph::rebindInput(Node *source, Node *input) {
  for (Channel *channel: input->outputs) {
    channel->source = source;
    source->outputs.push_back(channel);
  }
}

Node *Graph::rebindOutput(Node *output, Node *target) {
  assert(target->inputs.empty() &&
         "Cannot connect to DFVariable with existing inputs!");

  assert(output->inputs.size() == 1 && "Instance's output must be connected!");

  Channel *connect = output->inputs.front();
  Node *newSrc = connect->source;

  if (target->type == OpType::NONE) {
    // Find the ID of "connect" in "newSrcOutputs".
    // It is assumed the required connect exists in "newSrcOutputs".
    uint64_t connectId = 0;
    std::vector<Channel *> &newSrcOutputs = newSrc->outputs;
    for (; newSrcOutputs[connectId] != connect; ++connectId) {}

    // Remove existing (newSrc, output) connection.
    newSrcOutputs.erase(newSrcOutputs.begin() + connectId);

    // For each "target" node's output reconnect it to "newSrc".
    for (Channel *channel: target->outputs) {
      channel->source = newSrc;
      newSrcOutputs.push_back(channel);
    }

    // Old "target" is about to be deleted - change it for "newSrc".
    target = newSrc;
  } else if (target->type == OpType::OUT) {
    // Reconnect existing "newSrc" connection to "target".
    connect->target = target;
    target->inputs.push_back(connect);
  } else {
    assert(false && "DFVariables other than outputs and shallow are unsupported!");
  }

  return target;
}

void Graph::resetMeta(KernelMeta *meta) {
  for (Node *node: nodes) {
    node->var->setMeta(meta);
  }
}

} // namespace dfcxx
