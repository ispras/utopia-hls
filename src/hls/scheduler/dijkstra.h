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

struct PathNode final {
  PathNode() : nodeTime(0) {}
  
  std::vector<std::pair<Chan*, unsigned>> predessors;
  std::vector<std::pair<Chan*, unsigned>> successors;
  unsigned nodeTime;
};

class DijkstraBalancer final : public LatencyBalancer {
public:
  ~DijkstraBalancer();
  void balance(Model &model) override;

private:
  void reset();
  void deleteEntries();
  void init(const Graph *graph);
  void visit(PathNode *node);
  void insertBuffers(Model &model) override;

  std::deque<std::pair<const Chan*, unsigned>> toVisit;
  std::map<const Node*, PathNode*> nodeMap; 
};

} // namespace eda::hls::scheduler
