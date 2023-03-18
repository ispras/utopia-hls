//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "hls/mapper/mapper.h"
#include "hls/model/indicators.h"
#include "hls/model/model.h"
#include "hls/model/parameters.h"
#include "hls/scheduler/latency_solver.h"
#include "hls/scheduler/optimizers/abstract_optimizer.h"
#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"
#include "util/assert.h"
#include "util/singleton.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace eda::hls::scheduler {

template<typename T>
class ParametersOptimizer final : 
    public util::Singleton<ParametersOptimizer<T>> {
  
  friend class util::Singleton<ParametersOptimizer<T>>;

public:
  std::map<std::string, model::Parameters> optimize(
      const model::Criteria &criteria,
      model::Model &model,
      model::Indicators &indicators
  ) const;

  template <typename O>
  void set_optimizer(const O &optimizer) {
    math_optimizer = std::make_shared<O>(optimizer);
  }

private:
  ParametersOptimizer() = default;

  void estimate(model::Model &model, std::map<std::string, 
                model::Parameters> &params,
                model::Indicators &indicators,
                const std::vector<float> &optimized_params) const;

  void init(const model::Graph *graph, 
            std::map<std::string, model::Parameters> &parameters,
            std::vector<float> &parameterValues, std::vector<float> &minValues,
            std::vector<float> &maxValues) const;

  double normalize(double value, double min, double max) const;
  double denormalize(double value, double min, double max) const;

  std::shared_ptr<optimizers::AbstractOptimizer> math_optimizer;
};

#include "hls/scheduler/param_optimizer_impl.h"

} // namespace eda::hls::scheduler
