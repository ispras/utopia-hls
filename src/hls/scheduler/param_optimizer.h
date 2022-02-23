//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "hls/model/model.h"
#include "hls/scheduler/optimizers/abstract_optimizer.h"
#include "util/singleton.h"

#include <map>
#include <string>

using namespace eda::hls::library;
using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

struct Criteria final {
  Criteria(Indicator objective,
           const Constraint &frequency,
           const Constraint &throughput,
           const Constraint &latency,
           const Constraint &power,
           const Constraint &area):
    objective(objective),
    frequency(frequency),
    throughput(throughput),
    latency(latency),
    power(power),
    area(area) {}

  const Indicator objective;
  const Constraint frequency;
  const Constraint throughput;
  const Constraint latency;
  const Constraint power;
  const Constraint area;

  /// Checks the constraints.
  bool check(const Indicators &indicators) const {
    return frequency.check(indicators.frequency)
        && throughput.check(indicators.throughput)
        && latency.check(indicators.latency)
        && power.check(indicators.power)
        && area.check(indicators.area);
  }
};

class ParametersOptimizer final : public Singleton<ParametersOptimizer> {
  friend class Singleton<ParametersOptimizer>;

public:
  std::map<std::string, Parameters> optimize(
      const Criteria &criteria,
      Model &model,
      Indicators &indicators
  ) const;

  template <typename T>
  void set_optimizer(const T& optimizer) {
    math_optimizer = std::make_shared<T>(optimizer);
  }

private:
  ParametersOptimizer() = default;

  void count_params(Model& model, std::map<std::string, Parameters>& params,
                    Indicators& indicators, unsigned frequency, Parameters& defaultParams) const;

  std::shared_ptr<optimizers::abstract_optimizer> math_optimizer;
};

} // namespace eda::hls::scheduler
