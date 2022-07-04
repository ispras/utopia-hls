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

#pragma once

#include "hls/scheduler/latency_balancer_base.h"
#include "util/graph.h"
#include "util/singleton.h"

#include <unordered_set>

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

class TopologicalBalancer final : public TraverseBalancerBase, 
    public Singleton<TopologicalBalancer> {

public:
  friend Singleton<TopologicalBalancer>;
  ~TopologicalBalancer() {}
  void balance(Model &model) override;

private:
  TopologicalBalancer() {}

  void init(const Graph *graph);
  void visitNode(const Node *node) override;
  void visitChan(const Chan *chan) override;
  int getDelta(int curTime, const Chan* curChan) override;

  std::unordered_set<const Node*> visited;
  std::unordered_set<const Chan*> backEdges;
  const Node *currentNode;
};


} // namespace eda::hls::scheduler
