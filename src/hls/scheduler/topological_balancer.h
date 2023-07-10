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

namespace eda::hls::scheduler {

class TopologicalBalancer final : public TraverseBalancerBase, 
    public utils::Singleton<TopologicalBalancer> {

public:
  friend utils::Singleton<TopologicalBalancer>;
  ~TopologicalBalancer() {}
  void balance(model::Model &model) override;

private:
  TopologicalBalancer() {}

  void init(const model::Graph *graph);
  void visitNode(const model::Node *node) override;
  void visitChan(const model::Chan *chan) override;
  int getDelta(int curTime, const model::Chan* curChan) override;

  std::unordered_set<const model::Node*> visited;
  std::unordered_set<const model::Chan*> backEdges;
  const model::Node *currentNode;
};


} // namespace eda::hls::scheduler