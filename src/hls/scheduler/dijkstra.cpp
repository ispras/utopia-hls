//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/dijkstra.h"
#include "util/assert.h"

#include <iostream>

namespace eda::hls::scheduler {

void DijkstraBalancer::init(const Graph *graph) {
  nodeMap = std::map<const Node*, unsigned>();
  toVisit = std::deque<const Chan*>();
  terminalNodes = std::vector<const Node*>();

  // Init the elements
  for (const auto *node : graph->nodes) {
    nodeMap[node] = 0;
  }
}

void DijkstraBalancer::visitChan(const Chan *chan, unsigned targetTime) {
  const Node *targetNode = getNext(chan);
  // Update destination node time
  if (targetNode && targetTime > nodeMap[targetNode]) {
    nodeMap[targetNode] = targetTime;
  }
}

void DijkstraBalancer::visitSources(const Graph *graph) {
  for (const auto *chan : graph->chans) {
    const Node *src = chan->source.node;
    if (src->isSource() || src->isConst()) {
      toVisit.push_back(chan);
      visitChan(chan, nodeMap[src] + chan->ind.ticks);
    }
  }
}

void DijkstraBalancer::visitSinks(const Graph *graph) {
  for (const auto *chan : graph->chans) {
    const Node *targ = chan->target.node;
    if (targ->isSink()) {
      toVisit.push_back(chan);
      visitChan(chan, nodeMap[targ] + chan->ind.ticks);
    }
  }
}

const Node* DijkstraBalancer::getNext(const Chan *chan) {
  if (mode == LatencyBalanceMode::ASAP) {
    return chan->target.node;
  }

  if (mode == LatencyBalanceMode::ALAP) {
    return chan->source.node;
  }

  return nullptr;
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
  uassert(graph, "Graph 'main' not found!\n");
  init(graph);

  if (mode == LatencyBalanceMode::ASAP) {
    visitSources(graph);
  }

  if (mode == LatencyBalanceMode::ALAP) {
    visitSinks(graph);
  }

  while (!toVisit.empty()) {
    const Node *next = getNext(toVisit.front());
    if (next) {
      std::vector<Chan*> connections;
      addConnections(connections, next);
      for (const auto *chan : connections) {
        visitChan(chan, nodeMap[next] + chan->ind.ticks);
      }
    }
    toVisit.pop_front();
  }
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
        delta = curTime - (nodeMap[predNode] + pred->ind.ticks);
      }
      if (mode == LatencyBalanceMode::ALAP) {
        delta = (nodeMap[predNode] - pred->ind.ticks) - curTime;
      }
      uassert(delta >= 0,  "Delta for channel " + pred->name + " < 0!\n");
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
      maxTime = std::max(nodeMap[node], maxTime);
    }
    graphTime = maxTime;
    std::cout << "Max time: " << graphTime << std::endl;
}

} // namespace eda::hls::scheduler
