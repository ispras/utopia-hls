//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <hls/scheduler/scheduler.h>
#include <limits>
#include <map>
#include <queue>
#include <utility>
#include <vector>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

struct PathNode final {
  PathNode(unsigned startTime) : nodeTime(startTime) { }
  
  static unsigned getInitialValue(const Node* node) {
    return (node->type.name == "source") ? 0 : std::numeric_limits<unsigned>::infinity();
  }

  struct PathNodeCmp final {
    bool operator()(const PathNode* lhs, const PathNode* rhs) {
      return lhs->nodeTime < rhs->nodeTime;
    }
  };

  std::vector<std::pair<const Node*, unsigned>> successors;
  unsigned nodeTime;
};

class DijkstraBalancer final : public LatencyBalancer {
public:
  DijkstraBalancer(Model* modelArg) : LatencyBalancer(modelArg) { }
  ~DijkstraBalancer();
  void balance() override;

private:
  void reset();
  void deleteEntries();
  void init(const Graph* graph);
  void relax(const PathNode* src, std::pair<const Node*, unsigned> &dst);

  std::priority_queue<PathNode*, std::vector<PathNode*>, PathNode::PathNodeCmp> pathElements;
  std::vector<PathNode*> ready;
  std::map<const Node*, PathNode*> nodeMap; 

};

} // namespace eda::hls::scheduler
