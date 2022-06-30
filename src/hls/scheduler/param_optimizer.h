//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/indicators.h"
#include "hls/model/model.h"
#include "hls/model/parameters.h"
#include "hls/scheduler/latency_balancer_base.h"
#include "hls/scheduler/optimizers/abstract_optimizer.h"
#include "util/singleton.h"

#include <map>
#include <string>
#include <vector>

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

class ParametersOptimizer final : public Singleton<ParametersOptimizer> {
  friend class Singleton<ParametersOptimizer>;

public:
  std::map<std::string, Parameters> optimize(
      const Criteria &criteria,
      model::Model &model,
      Indicators &indicators
  ) const;

  template <typename T>
  void set_optimizer(const T &optimizer) {
    math_optimizer = std::make_shared<T>(optimizer);
  }

  void setBalancer(LatencyBalancerBase *latencyBalancer) {
    balancer = latencyBalancer;
  }

private:
  ParametersOptimizer() = default;

  void estimate(model::Model &model,
                std::map<std::string, Parameters> &params,
                Indicators &indicators,
                const std::vector<float> &optimized_params) const;

  void init(const Graph *graph, std::map<std::string, Parameters> &parameters, 
    std::vector<float> &parameterValues, std::vector<float> &minValues, 
    std::vector<float> &maxValues) const;

  double normalize(double value, double min, double max) const;
  double denormalize(double value, double min, double max) const;

  std::shared_ptr<optimizers::AbstractOptimizer> math_optimizer;

  LatencyBalancerBase *balancer;
};

} // namespace eda::hls::scheduler
