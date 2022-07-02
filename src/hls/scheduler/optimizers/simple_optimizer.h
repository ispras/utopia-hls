//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/scheduler/optimizers/abstract_optimizer.h"

#include <functional>
#include <vector>

namespace eda::hls::scheduler::optimizers {

class SimpleOptimizer
    : public eda::hls::scheduler::optimizers::AbstractOptimizer {
public:
  SimpleOptimizer(
      std::function<float(const std::vector<float> &)> targetFunction);
  SimpleOptimizer(const SimpleOptimizer &optimizer) = default;

  void optimize(std::vector<float> &parameterValues) override;

private:
  std::function<float(const std::vector<float> &)> targetFunction;

  float maxFrequency;
  float minFrequency;
  float limitation;
  float currentArea;
};

} // namespace eda::hls::scheduler::optimizers
