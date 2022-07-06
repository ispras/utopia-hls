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

class DijkstraBalancer final : public TraverseBalancerBase, 
    public Singleton<DijkstraBalancer> {
public:
  friend Singleton<DijkstraBalancer>;
  ~DijkstraBalancer() {}
  void balance(Model &model) override {
    balance(model, LatencyBalanceMode::ASAP);
  }
  
  void balance(Model &model, LatencyBalanceMode mode);

private:
  DijkstraBalancer() : mode(LatencyBalanceMode::ASAP) {}

  void init(const Graph *graph);

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

  std::queue<const Chan*> toVisit;
  LatencyBalanceMode mode;
  const Node *currentNode;
};

} // namespace eda::hls::scheduler
