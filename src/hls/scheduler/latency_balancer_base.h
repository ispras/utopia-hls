//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the declaration of the LatencyBalancerBase class, that
/// is the base class for all schedulers.
///
/// \author <a href="mailto:lebedev@ispras.ru">Mikhail Lebedev</a>
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"
#include "util/assert.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class LatencyBalancerBase {
public:
  virtual ~LatencyBalancerBase() {}

  /// Schedules the specified model.
  virtual void balance(eda::hls::model::Model &model) = 0;

  /// Returns the maximum latency of the main graph.
  unsigned getGraphLatency() { return graphTime; }

protected:
  LatencyBalancerBase() : graphTime(0) {}

  void init() {
    graphTime = 0;
  }

  /// Inserts the balancing buffers into the model.
  virtual void insertBuffers(eda::hls::model::Model &model) = 0;

  void printGraphTime() {
    std::cout << "Max time: " << graphTime << std::endl;
  }

  unsigned graphTime;
};

class TraverseBalancerBase : public LatencyBalancerBase {

public:
  
  virtual ~TraverseBalancerBase() {}

protected:
  TraverseBalancerBase() {}

  virtual void visitNode(const Node* node) = 0;
  virtual void visitChan(const Chan* chan) = 0;
  virtual int getDelta(int currentTime, const Chan *currentChan) = 0;

  // Init the elements
  void init(const Graph &graph) {
    LatencyBalancerBase::init();
    nodeMap.clear();
    for (const auto *node : graph.nodes) {
      nodeMap[node] = 0;
    }
  }

  void insertBuffers(Model &model) override {
    int bufsInserted = 0;
    int totalDelta = 0;
    for (const auto &node : nodeMap) {
      for (auto *currentChan : node.first->inputs) {
        int delta = getDelta(node.second, currentChan);
        uassert(delta >= 0,  
          "Delta for channel " + currentChan->name + " < 0!\n");
        if (delta > 0 && !currentChan->source.node->isConst()) {
          model.insertDelay(*currentChan, delta);
          bufsInserted++;
          totalDelta+=delta;
        }
      }
    }
    std::cout << "Total buffers inserted: " << bufsInserted << std::endl;
    std::cout << "Total buffers capacity: " << totalDelta << std::endl;
  }

  std::map<const Node*, int> nodeMap;
};

} // namespace eda::hls::scheduler
