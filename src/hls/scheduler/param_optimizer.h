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
           const Constraint &freq,
           const Constraint &perf,
           const Constraint &ticks,
           const Constraint &power,
           const Constraint &area):
    objective(objective),
    freq(freq),
    perf(perf),
    ticks(ticks),
    power(power),
    area(area) {}

  const Indicator objective;

  const Constraint freq;
  const Constraint perf;
  const Constraint ticks;
  const Constraint power;
  const Constraint area;

  /// Checks the constraints.
  bool check(const Indicators &indicators) const {
    return freq.check(indicators.freq())
        && perf.check(indicators.perf())
        && ticks.check(indicators.ticks)
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

  void estimate(Model& model, std::map<std::string, Parameters>& params,
                    Indicators& indicators, unsigned frequency) const;

  void updateFrequency(Model& model, std::map<std::string, Parameters>& params,
    const unsigned frequency) const;

  double normalize(double value, double min, double max) const;
  double denormalize(double value, double min, double max) const;

  std::shared_ptr<optimizers::abstract_optimizer> math_optimizer;
};

} // namespace eda::hls::scheduler
