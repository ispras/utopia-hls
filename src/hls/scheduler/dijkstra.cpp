//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/dijkstra.h"

#include <algorithm>
#include <iostream>

namespace eda::hls::scheduler {

void DijkstraBalancer::init(const Graph *graph) {
  TraverseBalancerBase::init(*graph);
  while (!toVisit.empty()) {
    toVisit.pop();
  }
  currentNode = nullptr;
}

void DijkstraBalancer::visitChan(const Chan *chan) {
  const Node *targetNode = getNextNode(chan);
  // Update destination node time
  if (targetNode && currentNode) {
    nodeMap[targetNode] = std::max(nodeMap[currentNode] + (int)chan->ind.ticks, 
      nodeMap[targetNode]);
    graphTime = std::max((unsigned)nodeMap[targetNode], graphTime);
  }
}

void DijkstraBalancer::visitNode(const Node *node) {
  currentNode = node;
  for (const auto *chan : getConnections(node)) {
    toVisit.push(chan);
    visitChan(chan);
  }
}

void DijkstraBalancer::start(const std::vector<Node*> &startNodes) {
  for (const auto *node : startNodes) {
    visitNode(node);
  }
}

const Node* DijkstraBalancer::getNextNode(const Chan *chan) {
  return 
    mode == LatencyBalanceMode::ASAP ? chan->target.node : chan->source.node;
}

const std::vector<Chan*>& DijkstraBalancer::getConnections(const Node *next) {
  return mode == LatencyBalanceMode::ASAP ? next->outputs : next->inputs;
}

void DijkstraBalancer::traverse(const std::vector<Node*> &startNodes) {
  start(startNodes);
  while (!toVisit.empty()) {
    visitNode(getNextNode(toVisit.front()));
    toVisit.pop();
  }
}

void DijkstraBalancer::balance(Model &model, LatencyBalanceMode balanceMode) {
  mode = balanceMode;
  const Graph *graph = model.main();
  uassert(graph, "Graph 'main' not found!\n");
  init(graph);

  if (mode == LatencyBalanceMode::ASAP) {
    traverse(graph->sources);
  }

  if (mode == LatencyBalanceMode::ALAP) {
    traverse(graph->targets);
  }

  insertBuffers(model);
  printGraphTime();
}

int DijkstraBalancer::getDelta(int curTime, const Chan* curChan) {
  // Compute delta between neighbouring nodes
  if (mode == LatencyBalanceMode::ASAP) {
    return curTime - (nodeMap[curChan->source.node] + (int)curChan->ind.ticks);
  }
  if (mode == LatencyBalanceMode::ALAP) {
    return (nodeMap[curChan->source.node] - (int)curChan->ind.ticks) - curTime;
  }
  return -1;
};

} // namespace eda::hls::scheduler
