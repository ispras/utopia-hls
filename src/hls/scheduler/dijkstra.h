//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/scheduler/scheduler.h"

#include <deque>
#include <map>
#include <utility>
#include <vector>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class DijkstraBalancer final : public LatencyBalancer {
public:
  ~DijkstraBalancer() {}
  void balance(Model &model) override;

private:
  void reset();
  void deleteEntries();
  void init(const Graph *graph);
  void visit(const Node *node);
  void insertBuffers(Model &model) override;

  std::deque<const Chan*> toVisit;
  std::map<const Node*, unsigned> nodeMap; 
};

} // namespace eda::hls::scheduler
