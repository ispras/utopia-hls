//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the declaration of the DijkstraBalancer class, that 
/// can schedule the dataflow graph using ASAP or ALAP techniques.
///
/// \author <a href="mailto:lebedev@ispras.ru">Mikhail Lebedev</a>
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/scheduler/latency_balancer_base.h"
#include "utils/singleton.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <queue>

namespace eda::hls::scheduler {

enum LatencyBalanceMode {
  /// Calculate the ASAP schedule
  ASAP,
  
  /// Calculate the ALAP schedule
  ALAP
};

class CompareChan final {
  std::unordered_map<const model::Node*, int> &nodeMap;
public:
  CompareChan(std::unordered_map<const model::Node*, int> &nodeMap) : 
      nodeMap(nodeMap) { }

  bool operator() (const model::Chan *lhs, const model::Chan *rhs) {
    return (nodeMap[lhs->source.node] + lhs->ind.ticks) < 
        (nodeMap[rhs->source.node] + rhs->ind.ticks);
  }
};

template <typename ElementType, typename Container, 
    typename Comparator = std::less<>>
class Queue {
public:

  Queue() {
    container = new Container();
  }

  Queue(Comparator comparator) {}

  ~Queue() {
    delete container;
  }

  void push(ElementType element) {
    return container->push(element);
  }

  void pop() {
    container->pop();
  }

  ElementType front() {
    return container->front();
  }
  
  bool empty() {
    return container->empty();
  }

  Container *container;
};

using StdPriorityQueue = std::priority_queue<const Chan*, 
    std::vector<const model::Chan*>, CompareChan>;
using GenericChanQueue = Queue<const model::Chan*, 
    StdPriorityQueue, CompareChan>;

template <typename Container, typename Comparator = std::less<>>
class DijkstraBalancer final : public TraverseBalancerBase, 
    public Singleton<DijkstraBalancer<Container, Comparator>> {
public:
  friend Singleton<DijkstraBalancer<Container, Comparator>>;

  ~DijkstraBalancer() {
    delete toVisit;
  }

  void balance(model::Model &model) override {
    balance(model, LatencyBalanceMode::ASAP);
  }
  
  void balance(model::Model &model, LatencyBalanceMode mode);

private:
  DijkstraBalancer() : mode(LatencyBalanceMode::ASAP) {}

  void init(const model::Graph *graph);

  void initQueue();

  /// Visits the specified channel and updates the destination's time.
  void visitChan(const model::Chan *chan) override;

  void visitNode(const model::Node *node) override;
  
  /// Returns the next node depending on the exploration direction: 
  /// ASAP - top->down
  /// ALAP - down->top
  const model::Node* getNextNode(const model::Chan *chan);

  /// Adds the connections of the node to the specified vector depending on 
  /// the exploration direction:
  /// ASAP - outputs
  /// ALAP - inputs
  const std::vector<model::Chan*>& getConnections(const model::Node *next);

  /// Visits the sources of the specified graph.
  void visitSources(const model::Graph *graph);

  /// Visits the sinks of the specified graph.
  void visitSinks(const model::Graph *graph);
  
  int getDelta(int curTime, const model::Chan* curChan) override;

  void start(const std::vector<model::Node*> &startNodes);

  void traverse(const std::vector<model::Node*> &startNodes);

  Queue<const model::Chan*, Container, Comparator> *toVisit = nullptr;
  LatencyBalanceMode mode;
  const model::Node *currentNode;
};

#include "hls/scheduler/dijkstra_impl.h"

} // namespace eda::hls::scheduler
