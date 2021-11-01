//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===

#include <cassert>
#include <hls/scheduler/scheduler.h>
#include <iostream>
#include <memory>

namespace eda::hls::scheduler {

LatencyBalancer::~LatencyBalancer() {
  for (auto* buf : buffers) {
    delete buf;
  }
}

void LatencyBalancer::insertBuffers(Graph* graph, 
    const std::vector<double> &latencies) {
  for (const auto* buf : buffers) {
    // lp_solve positions start with 1
    unsigned latency = latencies[buf->position - 1];
    if (latency != 0) {
      graph->insertDelay(*(buf->channel), latency);
      std::cout << "Inserted buffer: " << *(buf->channel)
                << " with latency " << latency << std::endl;
    }
  }
}

}  // namespace eda::hls::scheduler

