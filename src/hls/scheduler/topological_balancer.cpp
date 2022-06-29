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
    terminalNodes = std::unordered_set<const Node*>();
    visited = std::unordered_set<const Node*>();
    currentNode = nullptr;
  }

  void TopologicalBalancer::balance(Model &model) {
    reset();
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
    uassert(node, "Nullptr node found!\n");
    currentNode = node;
    if (node->isSink()) {
      terminalNodes.insert(node);
    }
    visited.insert(currentNode);
  }

  void TopologicalBalancer::visitChan(const Chan *chan) {
    uassert(chan, "Nullptr chan found!\n");
    if (currentNode) {
      const Node *targetNode = chan->target.node;
      uassert(targetNode, "Nullptr node found!");
      if (visited.find(targetNode) != visited.end()) {
        backEdges.insert(chan);
      } else {
        nodeMap[targetNode] = std::max(nodeMap[currentNode] + chan->ind.ticks, nodeMap[targetNode]);
      }
    }
  }
  
  void TopologicalBalancer::insertBuffers(Model &model) {
    unsigned bufsInserted = 0;
    for (const auto &targetNode : nodeMap) {
      const unsigned currentTime = targetNode.second;
      for (const auto &connection : targetNode.first->inputs) {
        const Node *sourceNode = connection->source.node;
        int delta = currentTime - (nodeMap[sourceNode] + connection->ind.ticks);
        if (backEdges.find(connection) != backEdges.end()) {
          delta = -delta;
        }

        uassert(delta >= 0,  "Delta for channel " + connection->name + " < 0!\n");
        if (delta > 0 && !sourceNode->isConst()) {
          model.insertDelay(*connection, delta);
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
