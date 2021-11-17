//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/scheduler/scheduler.h"
#include "util/singleton.h"

#include <deque>
#include <map>
#include <utility>
#include <vector>

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

class DijkstraBalancer final : public LatencyBalancer, 
    public Singleton<DijkstraBalancer> {
public:
  friend Singleton<DijkstraBalancer>;
  ~DijkstraBalancer() {}
  void balance(Model &model) override {
    balance(model, BalanceMode::LatencyASAP);
  }
  
  void balance(Model &model, BalanceMode mode);

private:
  DijkstraBalancer() : mode(BalanceMode::LatencyASAP) {}
  void reset();
  void deleteEntries();
  void init(const Graph *graph);
  void visit(unsigned curTime, const std::vector<Chan*> &connections);
  void visitChan(const Chan *chan, unsigned dstTime);
  const Node* getNext(const Chan *chan);
  void addConnections(std::vector<Chan*> &connections, const Node *next);
  void collectSources(const Graph *graph);
  void collectSinks(const Graph *graph);
  void insertBuffers(Model &model) override;
  void collectGraphTime() override;

  std::deque<const Chan*> toVisit;
  std::map<const Node*, unsigned> nodeMap;
  BalanceMode mode;
  std::vector<const Node*> terminalNodes;
};

} // namespace eda::hls::scheduler
