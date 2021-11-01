//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/dijkstra.h"

namespace eda::hls::scheduler {

DijkstraBalancer::~DijkstraBalancer() {
  deleteEntries();
}

void DijkstraBalancer::deleteEntries() {
  for (; !pathElements.empty(); pathElements.pop()) {
    delete pathElements.top();
  }

  for (auto* node : ready) {
    delete node;
  }

  for (auto entry : nodeMap) {
    delete entry.second;
  }
}

void DijkstraBalancer::reset() {

  deleteEntries();  

  // Reset the queues
  pathElements = std::priority_queue<PathNode*, std::vector<PathNode*>, 
      PathNode::PathNodeCmp>();
  ready = std::vector<PathNode*>();
  nodeMap = std::map<const Node*, PathNode*>();
}

void DijkstraBalancer::init(const Graph* graph) {
  reset();
  // Init the elements
  for (const auto *node : graph->nodes) {
    PathNode *pNode = new PathNode(PathNode::getInitialValue(*node));
    pathElements.push(pNode);
    nodeMap[node] = pNode;
    for (const auto *chan : node->outputs) {
      unsigned latency = chan->source.port->latency;
      pNode->successors.push_back(std::make_pair(chan->target.node, latency));
    }
  }
}

void DijkstraBalancer::relax(const PathNode *src,
    std::pair<const Node*, unsigned> &dst) {
  unsigned curTime = src->nodeTime;
  PathNode *dstNode = nodeMap[dst.first];
  unsigned dstTime = curTime + dst.second;

  if (dstTime < dstNode->nodeTime) {
    dstNode->nodeTime = dstTime;
  }
}

void DijkstraBalancer::balance(Model &model) {
  for (const auto *graph : model.graphs) {
    init(graph);
    for (; !pathElements.empty(); pathElements.pop()) {
      PathNode *node = pathElements.top();
      for (auto succ : node->successors) {
        relax(node, succ);
      }
      ready.push_back(node);
    }
  }
}

} // namespace eda::hls::scheduler
