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
  LatencyLinear,
  LatencyDijkstra
};

struct Buffer;

class LatencyBalancer {
public:
  LatencyBalancer() { }
  LatencyBalancer(Model* model_arg) : model(model_arg) { }
  ~LatencyBalancer() {
    for (auto buf : buffers) {
      delete buf;
    }
  }
  void setModel(Model* model_arg) { model = model_arg; }
  void balance();

protected:
  void insertBuffers(const Graph* graph, const std::vector<double> &latencies);

  Model* model;
  std::vector<Buffer*> buffers;
};

struct Buffer final {

  /*Buffer(const Node* src, const Node* dst, const Chan* chan, 
      const SolverVariable* var) : source(src), destination(dst), channel(chan),
      variable(var) { }*/

  const Node* source;
  const Node* destination;
  const Chan* channel;
  //const SolverVariable* variable;
};

} // namespace eda::hls::scheduler
