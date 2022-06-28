//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
///
/// \author <a href="mailto:lebedev@ispras.ru">Mikhail Lebedev</a>
//===----------------------------------------------------------------------===//

#include "hls/scheduler/topological_balancer.h"
#include "util/assert.h"

#include <iostream>

namespace eda::hls::scheduler {

  void TopologicalBalancer::reset() {
    nodeMap = std::map<const Node*, unsigned>();
    terminalNodes = std::vector<const Node*>();
  }

  void TopologicalBalancer::balance(Model &model) {
    Graph *graph = model.main();
    uassert(graph, "Graph 'main' not found!\n");

    for (const auto *node : graph->nodes) {
      nodeMap[node] = 0;
    }

    auto handleNode = [this](const Node *node) { visitNode(node); };
    auto handleEdge = [this](const Chan *chan) { visitChan(chan); };

    graph::traverseTopologicalOrder<Graph, const Node*, const Chan*>(*graph, handleNode, handleEdge);
    insertBuffers(model);
    collectGraphTime();
  }

  void TopologicalBalancer::visitNode(const Node *node) {
    currentNode = node;
    std::cout << "Visiting node: " << node->name << std::endl;
  }

  void TopologicalBalancer::visitChan(const Chan *chan) {
    //std::cout << "Visiting chan: " << chan->name << std::endl;
    if (currentNode) {
      const Node *targetNode = chan->target.node;
      if (targetNode && (nodeMap[currentNode] + chan->ind.ticks > nodeMap[targetNode])) {
        nodeMap[targetNode] = nodeMap[currentNode] + chan->ind.ticks;
        /*std::cout << "Node: " << targetNode->name << " Time update: " << nodeMap[currentNode] + chan->ind.ticks << std::endl;
        std::cout << "Pred node: " << currentNode->name << std::endl;
        std::cout << "Chan: " << chan->name << std::endl << std::endl;*/
      }
    }
  }
  
  void TopologicalBalancer::insertBuffers(Model &model) {
    unsigned bufsInserted = 0;
    for (const auto &node : nodeMap) {
      const unsigned curTime = node.second;
      for (const auto &pred : node.first->inputs) {
        const Node *predNode = pred->source.node;
        int delta = curTime - (nodeMap[predNode] + pred->ind.ticks);
        /*std::cout << "Node: " << node.first -> name << " Time: " << curTime << std::endl;
        std::cout << "Pred node: " << predNode->name << " Pred time: " << nodeMap[predNode] 
          << " Ticks: " << pred->ind.ticks << std::endl;
        std::cout << "Delta: " << delta << std::endl;
        std::cout << std::endl;*/

        uassert(delta >= 0,  "Delta for channel " + pred->name + " < 0!\n");
        if (delta > 0 && !predNode->isConst()) {
          model.insertDelay(*pred, delta);
          bufsInserted++;
        }
      }
    }
    std::cout << "Total buffers inserted: " << bufsInserted << std::endl;
  }

  void TopologicalBalancer::collectGraphTime() {
    unsigned maxTime = 0;
    for (const auto *node : terminalNodes) {
      maxTime = std::max(nodeMap[node], maxTime);
    }
    graphTime = maxTime;
    std::cout << "Max time: " << graphTime << std::endl;
  }

} // namespace eda::hls::scheduler
