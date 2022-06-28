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

#include <map>

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

class TopologicalBalancer : public LatencyBalancerBase, 
    public Singleton<TopologicalBalancer> {

public:
  friend Singleton<TopologicalBalancer>;
  ~TopologicalBalancer() {}
  void balance(Model &model) override;

private:
  TopologicalBalancer() {}

  void reset();
  void visitNode(const Node *node);
  void visitChan(const Chan *chan);
  void insertBuffers(Model &model) override;
  void collectGraphTime() override;

  std::map<const Node*, unsigned> nodeMap;
  std::vector<const Node*> terminalNodes;
  const Node *currentNode;

};


} // namespace eda::hls::scheduler
