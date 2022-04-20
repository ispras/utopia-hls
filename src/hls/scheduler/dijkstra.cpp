//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/dijkstra.h"

#include <cassert>
#include <iostream>

namespace eda::hls::scheduler {

void DijkstraBalancer::reset() {
   nodeMap = std::map<const Node*, unsigned>();
   toVisit = std::deque<const Chan*>();
   terminalNodes = std::vector<const Node*>();
}

void DijkstraBalancer::init(const Graph *graph) {
  reset();
  // Init the elements
  for (const auto *node : graph->nodes) {
    nodeMap[node] = 0;
  }
}

void DijkstraBalancer::visitChan(const Chan *chan, unsigned dstTime) {
  const Node *dstNode = getNext(chan);
  // Update destination node time
  if (dstNode != nullptr && dstTime > nodeMap[dstNode]) {
    nodeMap[dstNode] = dstTime;
  }
}

void DijkstraBalancer::visit(unsigned curTime, 
    const std::vector<Chan*> &connections) {
  // Visit neighbours and update their times
  for (const auto *next : connections) {
    visitChan(next, curTime + next->source.port->latency);
  }
}

void DijkstraBalancer::visitSources(const Graph *graph) {
  for (const auto *chan : graph->chans) {
    const Node *src = chan->source.node;
    if (src->isSource()) {
      toVisit.push_back(chan);
      visitChan(chan, nodeMap[src] + chan->source.port->latency);
    }
  }
}

void DijkstraBalancer::visitSinks(const Graph *graph) {
  for (const auto *chan : graph->chans) {
    const Node *targ = chan->target.node;
    if (targ->isSink()) {
      toVisit.push_back(chan);
      visitChan(chan, nodeMap[targ] + chan->source.port->latency);
    }
  }
}

const Node* DijkstraBalancer::getNext(const Chan *chan) {
  return mode == LatencyBalanceMode::ASAP ? chan->target.node :
    mode == LatencyBalanceMode::ALAP ? chan->source.node : nullptr;
}

void DijkstraBalancer::addConnections(std::vector<Chan*> &connections, 
    const Node *next) {
  if (mode == LatencyBalanceMode::ASAP) connections = next->outputs;
  if (mode == LatencyBalanceMode::ALAP) connections = next->inputs;
  // Add neighbours to the queue
  toVisit.insert(toVisit.end(), connections.begin(), connections.end());
  // Collect sinks or sources
  if (connections.empty()) {
    terminalNodes.push_back(next);
  }
}

void DijkstraBalancer::balance(Model &model, LatencyBalanceMode balanceMode) {
  mode = balanceMode;
  const Graph *graph = model.main();
  init(graph);

  if (mode == LatencyBalanceMode::ASAP) {
    visitSources(graph);
  }

  if (mode == LatencyBalanceMode::ALAP) {
    visitSinks(graph);
  }

  while (!toVisit.empty()) {
    const Node *next = getNext(toVisit.front());
    std::vector<Chan*> connections;
    addConnections(connections, next);
    if (next != nullptr) {
      visit(nodeMap[next], connections);
    }
    toVisit.pop_front();
  }
  /*for (const auto &elem : nodeMap) {
    std::cout << elem.first->name << " ";
    std::cout << elem.second << std::endl;
  }*/
  insertBuffers(model);
  collectGraphTime();
}

void DijkstraBalancer::insertBuffers(Model &model) {
  unsigned bufsInserted = 0;
  for (const auto &node : nodeMap) {
    const unsigned curTime = node.second;
    for (const auto &pred : node.first->inputs) {
      const Node *predNode = pred->source.node;
      int delta = 0;
      // Compute delta between neighbouring nodes
      if (mode == LatencyBalanceMode::ASAP) {
        delta = curTime - (nodeMap[predNode] + pred->source.port->latency);
      }
      if (mode == LatencyBalanceMode::ALAP) {
        delta = (nodeMap[predNode] - pred->source.port->latency) - curTime;
      }
      assert(delta >= 0 && ("Delta for channel " + pred->name + " < 0").c_str());
      if (delta > 0 && !predNode->isConst()) {
        model.insertDelay(*pred, delta);
        bufsInserted++;
      }
    }
  }
  std::cout << "Total buffers inserted: " << bufsInserted << std::endl;
}

void DijkstraBalancer::collectGraphTime() {
  unsigned maxTime = 0;
  for (const auto *node : terminalNodes) {
    unsigned nodeTime = nodeMap[node];
    if (nodeTime > maxTime) {
      maxTime = nodeTime;
    }
  }
  graphTime = maxTime;
  std::cout << "Max time: " << graphTime << std::endl;
}

} // namespace eda::hls::scheduler
