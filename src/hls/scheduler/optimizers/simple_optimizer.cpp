//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/optimizers/simple_optimizer.h"

namespace eda::hls::scheduler::optimizers {
SimpleOptimizer::SimpleOptimizer(
    std::function<float(const std::vector<float> &)> targetFunction)
    : targetFunction(targetFunction) {}

void SimpleOptimizer::optimize(std::vector<float> &parameterValues) {
  int y1, y2;
  int x2 = parameterValues[0];
  y2 = targetFunction(parameterValues);
  int x1 = maxFrequency - (maxFrequency - minFrequency) / 10;
  targetFunction(parameterValues);
  y1 = currentArea;
  float k = float(y1 - y2) / float(x1 - x2);
  float b = float(y1 - x1 * k);
  parameterValues[0] = (limitation - b) / k;
  targetFunction(parameterValues);
  int sign;
  if (currentArea > limitation) {
    sign = -1;
  } else {
    sign = 1;
  }
  int grid = (maxFrequency - minFrequency) / 100;

  // Optimization loop.
  const unsigned N = 5;
  for (unsigned i = 0; i < N; i++) {

    parameterValues[0] += sign * grid;
    currentArea = targetFunction(parameterValues);
    // Check the constraints.
    if (currentArea < limitation) {
      break;
    }
  }
}

} // namespace eda::hls::scheduler::optimizers