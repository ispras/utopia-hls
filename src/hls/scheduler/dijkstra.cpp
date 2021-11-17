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
   toVisit = std::deque<const Chan*>();
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
  if (dstNode != nullptr && dstTime > nodeMap[dstNode]) {
    nodeMap[dstNode] = dstTime;
  }
}

void DijkstraBalancer::visit(unsigned curTime, 
    const std::vector<Chan*> &connections) {
  for (const auto *next : connections) {
    visitChan(next, curTime + next->source.port->latency);
  }
}

void DijkstraBalancer::collectSources(const Graph *graph) {
  for (const auto *chan : graph->chans) {
    const Node *src = chan->source.node;
    if (src->isSource()) {
      toVisit.push_back(chan);
      visitChan(chan, nodeMap[src] + chan->source.port->latency);
    }
  }
}

void DijkstraBalancer::collectSinks(const Graph *graph) {
  for (const auto *chan : graph->chans) {
    const Node *targ = chan->target.node;
    if (targ->isSink()) {
      toVisit.push_back(chan);
      visitChan(chan, nodeMap[targ] + chan->source.port->latency);
    }
  }
}

const Node* DijkstraBalancer::getNext(const Chan *chan) {
  return mode == BalanceMode::LatencyASAP ? chan->target.node :
    mode == BalanceMode::LatencyALAP ? chan->source.node : nullptr;
}

void DijkstraBalancer::addConnections(std::vector<Chan*> &connections, 
    const Node *next) {
  if (mode == BalanceMode::LatencyASAP) connections = next->outputs;
  if (mode == BalanceMode::LatencyALAP) connections = next->inputs;
  toVisit.insert(toVisit.end(), connections.begin(), connections.end());
  if (connections.empty()) {
    terminalNodes.push_back(next);
  }
}

void DijkstraBalancer::balance(Model &model, BalanceMode balanceMode) {
  mode = balanceMode;
  for (const auto *graph : model.graphs) {
    if (graph->isMain()) {
      init(graph);

      if (mode == BalanceMode::LatencyASAP) {
        collectSources(graph);
      }

      if (mode == BalanceMode::LatencyALAP) {
        collectSinks(graph);
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
    }
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
      if (mode == BalanceMode::LatencyASAP) {
        delta = curTime - (nodeMap[predNode] + pred->source.port->latency);
      }
      if (mode == BalanceMode::LatencyALAP) {
        delta = (nodeMap[predNode] - pred->source.port->latency) - curTime;
      }
      assert(delta >= 0);
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
