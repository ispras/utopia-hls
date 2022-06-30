#pragma once

#include "hls/scheduler/optimizers/abstract_optimizer.h"
#include "hls/scheduler/param_optimizer.h"

#include <functional>

namespace eda::hls::scheduler::optimizers {

class SimulatedAnnealingOptimizer final : public AbstractOptimizer {

public:
  using target = std::function<float(const std::vector<float> &)>;
  using condition = std::function<float(const std::vector<float> &)>;
  using step = std::function<void(std::vector<float> &,
                                  const std::vector<float> &, float, float)>;
  using temperature = std::function<float(int, float)>;

  SimulatedAnnealingOptimizer(float initialTemperature, float finalTemperature,
                              target targetFunction,
                              condition conditionFunction, step stepFunction,
                              temperature temperatureFunction);
  SimulatedAnnealingOptimizer(const SimulatedAnnealingOptimizer &optimizer) =
      default;

  void optimize(std::vector<float> &param) override;

private:
  float getProbability(float previousTarget, float currentTarget,
                       float currentTemperature);

  float currentTemperature;
  float initialTemperature;
  float finalTemperature;

  target targetFunction;
  condition conditionFunction;
  step stepFunction;
  temperature temperatureFunction;
};

} // namespace eda::hls::scheduler::optimizers
