//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <hls/model/model.h>
#include <string>
#include <vector>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

enum BalanceMode {
  Simple,
  Blocking,
  LatencyLP,
  LatencyDijkstra
};

struct Buffer;

class LatencyBalancer {
public:
  LatencyBalancer() { }
  LatencyBalancer(Model* modelArg) : model(modelArg) { }
  virtual ~LatencyBalancer();
  void setModel(Model* modelArg) { model = modelArg; }
  Model* getModel() { return model; }
  virtual void balance() { }

protected:
  void insertBuffers(Graph* graph, const std::vector<double> &latencies);

  Model* model;
  std::vector<Buffer*> buffers;
};

struct Buffer final {
  Buffer(Chan* chan, unsigned latency, unsigned position) : channel(chan), 
      latency(latency), position(position) { }

  Chan* channel;
  unsigned latency;
  unsigned position;
  //const SolverVariable* variable;
};

} // namespace eda::hls::scheduler
