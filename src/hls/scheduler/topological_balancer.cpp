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

#include <algorithm>

namespace eda::hls::scheduler {

  void TopologicalBalancer::init(const model::Graph *graph) {
    TraverseBalancerBase::init(*graph);
    visited.clear();
    currentNode = nullptr;
  }

  void TopologicalBalancer::balance(model::Model &model) {
    model::Graph *graph = model.main();
    uassert(graph, "Graph 'main' not found!\n");
    init(graph);

    auto handleNode = [this](const model::Node *node) { visitNode(node); };
    auto handleEdge = [this](const model::Chan *chan) { visitChan(chan); };

    graph::traverseTopologicalOrder<model::Graph>(*graph, handleNode, handleEdge);
    insertBuffers(model);
    printGraphTime();
  }

  void TopologicalBalancer::visitNode(const model::Node *node) {
    uassert(node, "Nullptr node found!\n");
    currentNode = node;
    visited.insert(currentNode);
  }

  void TopologicalBalancer::visitChan(const model::Chan *chan) {
    uassert(chan, "Nullptr chan found!\n");
    if (currentNode) {
      const model::Node *targetNode = chan->target.node;
      if (visited.find(targetNode) != visited.end()) {
        backEdges.insert(chan);
      } else {
        nodeMap[targetNode] = std::max(nodeMap[currentNode] + 
          (int)chan->ind.ticks, nodeMap[targetNode]);
        graphTime = std::max((unsigned)nodeMap[targetNode], graphTime);
      }
    }
  }

  int TopologicalBalancer::getDelta(int curTime, const model::Chan* curChan) {
    // FIXME: feedback buffer size
    return (backEdges.find(curChan) == backEdges.end()) ? 
      curTime - (nodeMap[curChan->source.node] + (int)curChan->ind.ticks) : 0;
  };
   
} // namespace eda::hls::scheduler
