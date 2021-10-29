//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===

#include <cassert>
#include <hls/scheduler/scheduler.h>
#include <memory>

namespace eda::hls::scheduler {

LatencyBalancer::~LatencyBalancer() {
    for (auto* buf : buffers) {
      delete buf;
    }
  }

void LatencyBalancer::insertBuffers(const Graph* graph, 
    const std::vector<double> &latencies) {
  /*for (auto const buf : buffers) {
    if (latencies[buf->variable->column_number] != 0.0) {
      NodeType* type = findType("delay");
      assert(type != nullptr);
      Node* delay = new Node{buf->variable->name, *type, *graph};
      const Chan* preDelay = buf->channel;
      (preDelay->target.node) = delay;
      Chan* postDelay = new Chan{}
    }
  }*/
}

}  // namespace eda::hls::scheduler

