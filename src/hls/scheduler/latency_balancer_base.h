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
#include "utils/assert.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace eda::hls::scheduler {

class LatencyBalancerBase {
public:
  virtual ~LatencyBalancerBase() {}

  /// Schedules the specified model.
  virtual void balance(model::Model &model) = 0;

  /// Returns the maximum latency of the main graph.
  unsigned getGraphLatency() const { return graphTime; }

protected:
  LatencyBalancerBase() : graphTime(0) {}

  void init() {
    graphTime = 0;
  }

  /// Inserts the balancing buffers into the model.
  virtual void insertBuffers(model::Model &model) = 0;

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

  virtual void visitNode(const model::Node *node) = 0;
  virtual void visitChan(const model::Chan *chan) = 0;
  virtual int getDelta(int currentTime, 
      const model::Chan *currentChan) = 0;

  // Init the elements
  void init(const model::Graph &graph) {
    LatencyBalancerBase::init();
    nodeMap.clear();
    for (const auto *node : graph.nodes) {
      nodeMap[node] = 0;
    }
  }

  void insertBuffers(model::Model &model) override {
    int bufsInserted = 0;
    int totalDelta = 0;
    for (const auto &[node, time] : nodeMap) {
      for (auto *currentChan : node->inputs) {
        int delta = getDelta(time, currentChan);
        uassert(delta >= 0,  
          "Delta for channel " + currentChan->name + " < 0!\n");
        if (delta > 0 && !currentChan->source.node->isConst()) {
          currentChan->latency = delta;
          model.insertDelay(*currentChan, delta);
          bufsInserted++;
          totalDelta+=delta;
        }
      }
    }
    std::cout << "Total buffers inserted: " << bufsInserted << std::endl;
    std::cout << "Total buffers capacity: " << totalDelta << std::endl;
  }

  std::unordered_map<const model::Node*, int> nodeMap;
};

} // namespace eda::hls::scheduler
