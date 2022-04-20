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

#include <deque>
#include <map>
#include <utility>
#include <vector>

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

enum LatencyBalanceMode {
  /// Calculate the ASAP schedule
  ASAP,
  
  /// Calculate the ALAP schedule
  ALAP
};

class DijkstraBalancer final : public LatencyBalancerBase, 
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
  void reset();
  void init(const Graph *graph);

  /// Visits the specified channels.
  void visit(unsigned curTime, const std::vector<Chan*> &connections);
  
  /// Visits the specified channel and updates the destination's time.
  void visitChan(const Chan *chan, unsigned dstTime);
  
  /// Returns the next node depending on the exploration direction: 
  /// ASAP - top->down
  /// ALAP - down->top
  const Node* getNext(const Chan *chan);

  /// Adds the connections of the node to the specified vector depending on 
  /// the exploration direction:
  /// ASAP - outputs
  /// ALAP - inputs
  void addConnections(std::vector<Chan*> &connections, const Node *next);

  /// Visits the sources of the specified graph.
  void visitSources(const Graph *graph);

  /// Visits the sinks of the specified graph.
  void visitSinks(const Graph *graph);
  
  void insertBuffers(Model &model) override;
  void collectGraphTime() override;

  std::deque<const Chan*> toVisit;
  std::map<const Node*, unsigned> nodeMap;
  LatencyBalanceMode mode;
  std::vector<const Node*> terminalNodes;
};

} // namespace eda::hls::scheduler
