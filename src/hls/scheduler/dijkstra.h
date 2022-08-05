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
#include "util/singleton.h"

#include <algorithm>
#include <iostream>
#include <queue>

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

enum LatencyBalanceMode {
  /// Calculate the ASAP schedule
  ASAP,
  
  /// Calculate the ALAP schedule
  ALAP
};

class CompareChan final {
  std::unordered_map<const Node*, int> &nodeMap;
public:
  CompareChan(std::unordered_map<const Node*, int> &nodeMap) : nodeMap(nodeMap) { }

  bool operator() (const Chan *lhs, const Chan *rhs) {
    return (nodeMap[lhs->source.node] + lhs->ind.ticks) < (nodeMap[rhs->source.node] + rhs->ind.ticks);
  }
};

template <typename T, typename C>
class Queue {
public:

  Queue() {
    container = new C();
  }

  Queue(void *comparator) { }

  ~Queue() {
    delete container;
  }

  void push(T element) {
    return container->push(element);
  }

  void pop() {
    container->pop();
  }

  T front() {
    return container->front();
  }
  
  bool empty() {
    return container->empty();
  }

  C *container;
};

using StdPriorityQueue = std::priority_queue<const Chan*, std::vector<const Chan*>, CompareChan>;
using GenericChanQueue = Queue<const Chan*, StdPriorityQueue>;

template <typename C>
class DijkstraBalancer final : public TraverseBalancerBase, 
    public Singleton<DijkstraBalancer<C>> {
public:
  friend Singleton<DijkstraBalancer<C>>;

  ~DijkstraBalancer() {
    delete toVisit;
  }

  void balance(Model &model) override {
    balance(model, LatencyBalanceMode::ASAP);
  }
  
  void balance(Model &model, LatencyBalanceMode mode);

private:
  DijkstraBalancer() : mode(LatencyBalanceMode::ASAP) {}

  void init(const Graph *graph);

  void initQueue();

  /// Visits the specified channel and updates the destination's time.
  void visitChan(const Chan *chan) override;

  void visitNode(const Node *node) override;
  
  /// Returns the next node depending on the exploration direction: 
  /// ASAP - top->down
  /// ALAP - down->top
  const Node* getNextNode(const Chan *chan);

  /// Adds the connections of the node to the specified vector depending on 
  /// the exploration direction:
  /// ASAP - outputs
  /// ALAP - inputs
  const std::vector<Chan*>& getConnections(const Node *next);

  /// Visits the sources of the specified graph.
  void visitSources(const Graph *graph);

  /// Visits the sinks of the specified graph.
  void visitSinks(const Graph *graph);
  
  int getDelta(int curTime, const Chan* curChan) override;

  void start(const std::vector<Node*> &startNodes);

  void traverse(const std::vector<Node*> &startNodes);

  Queue<const Chan*, C> *toVisit = nullptr;
  LatencyBalanceMode mode;
  const Node *currentNode;
};

#include "hls/scheduler/dijkstra_impl.h"

} // namespace eda::hls::scheduler
