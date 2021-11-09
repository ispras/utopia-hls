//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/dijkstra.h"

#include <cassert>
#include <iostream>

namespace eda::hls::scheduler {

void DijkstraBalancer::reset() {
   nodeMap = std::map<const Node*, unsigned>();
}

void DijkstraBalancer::init(const Graph* graph) {
  reset();
  // Init the elements
  for (const auto *node : graph->nodes) {
    nodeMap[node] = 0;
  }
}

void DijkstraBalancer::visit(const Node *node) {

  toVisit.insert(toVisit.end(), node->outputs.begin(), node->outputs.end());

  for (const auto &next : node->outputs) {
    unsigned curTime = nodeMap[node];
    const Node *dstNode = next->target.node;
    unsigned dstTime = curTime + next->source.port->latency;
    if (dstTime > nodeMap[dstNode]) {
      nodeMap[dstNode] = dstTime;
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
          toVisit.push_back(chan);
        }
      }

      while (!toVisit.empty()) {
        visit(toVisit.front()->target.node);
        toVisit.pop_front();
      }
    }
  }
  insertBuffers(model);
}

void DijkstraBalancer::insertBuffers(Model &model) {
  unsigned bufsInserted = 0;
  for (const auto &node : nodeMap) {
    const unsigned curTime = node.second;
    for (const auto &pred : node.first->inputs) {
      const Node *predNode = pred->source.node;
      int delta = curTime - (nodeMap[predNode] + pred->source.port->latency);
      assert(delta >= 0);
      if (delta > 0) {
        model.insertDelay(*pred, delta);
        bufsInserted++;
      }
    }
  }
  std::cout << "Total buffers inserted: " << bufsInserted << std::endl;
}

} // namespace eda::hls::scheduler
