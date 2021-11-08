//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/dijkstra.h"

#include <cassert>

namespace eda::hls::scheduler {

void addNeighbours(std::vector<std::pair<Chan*, unsigned>>&, const std::vector<Chan*>&);

DijkstraBalancer::~DijkstraBalancer() {
  deleteEntries();
}

void DijkstraBalancer::deleteEntries() {
  for (auto entry : nodeMap) {
    delete entry.second;
  }
}

void DijkstraBalancer::reset() {
  deleteEntries();  
  nodeMap = std::map<const Node*, PathNode*>();
}

void addNeighbours(std::vector<std::pair<Chan*, unsigned>> &dest, 
    const std::vector<Chan*> &neighbours) {
  for (auto *chan : neighbours) {
    unsigned latency = chan->source.port->latency;
    dest.push_back(std::make_pair(chan, latency));
  }
}

void DijkstraBalancer::init(const Graph* graph) {
  reset();
  // Init the elements
  for (const auto *node : graph->nodes) {
    PathNode *pNode = new PathNode;
    nodeMap[node] = pNode;
    
    addNeighbours(pNode->successors, node->outputs);
    addNeighbours(pNode->predessors, node->inputs);
  }
}

void DijkstraBalancer::visit(PathNode *node) {

  toVisit.insert(toVisit.end(), node->successors.begin(), node->successors.end());

  for (const auto &next : node->successors) {
    unsigned curTime = node->nodeTime;
    PathNode *dstNode = nodeMap[next.first->target.node];
    unsigned dstTime = curTime + next.second;
    if (dstTime > dstNode->nodeTime) {
      dstNode->nodeTime = dstTime;
    }
  }
}

void DijkstraBalancer::balance(Model &model) {
  for (const auto *graph : model.graphs) {
    if (graph->isMain()) {
      init(graph);

      for (const auto *chan : graph->chans) {
        const Node *src = chan->source.node;
        if (src->isSource()) {
          toVisit.push_back(std::make_pair(chan, chan->source.port->latency));
        }
      }

      while (!toVisit.empty()) {
        visit(nodeMap[toVisit.front().first->target.node]);
        toVisit.pop_front();
      }
    }
  }
  insertBuffers(model);
}

void DijkstraBalancer::insertBuffers(Model &model) {

  for (const auto &node : nodeMap) {
    const unsigned curTime = node.second->nodeTime;
    for (const auto &pred : node.second->predessors) {
      const PathNode *predNode = nodeMap[pred.first->source.node];
      int delta = curTime - (predNode->nodeTime + pred.second);
      assert(delta >= 0);
      if (delta > 0) {
        model.insertDelay(*pred.first, delta);
      }
    }
  }

}

} // namespace eda::hls::scheduler
