//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace eda::hls::scheduler::optimizers {
SimulatedAnnealingOptimizer::SimulatedAnnealingOptimizer(
    float initialTemperature, float finalTemperature, target targetFunction,
    condition conditionFunction, step stepFunction,
    temperature temperatureFunction)
    : currentTemperature(initialTemperature),
      initialTemperature(initialTemperature),
      finalTemperature(finalTemperature), targetFunction(targetFunction),
      conditionFunction(conditionFunction), stepFunction(stepFunction),
      temperatureFunction(temperatureFunction) {}

void SimulatedAnnealingOptimizer::optimize(std::vector<float> &currentValues) {
  std::vector<float> candidateValues = currentValues;
  float currentTarget = -1000000.0, candidateTarget, transitionProbability;

  std::ofstream ostrm("annealing.txt");
  int i = 1;
  while (currentTemperature > finalTemperature && i < 10000) {
    candidateTarget = targetFunction(candidateValues);
    auto currentLimit = conditionFunction(candidateValues);
    transitionProbability =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    auto stepProbability =
        getProbability(currentTarget, candidateTarget, currentTemperature);
    ostrm << "Temperature: " << currentTemperature << std::endl;
    ostrm << "Transition: " << transitionProbability << std::endl;
    ostrm << "Probability: " << stepProbability << std::endl;
    ostrm << "Best target: " << currentTarget << std::endl;
    ostrm << "Current target: " << candidateTarget << std::endl;
    ostrm << "Current limitation: " << currentLimit << std::endl;
    ostrm << "Current params: " << candidateValues[0] << " "
          << candidateValues[1] << std::endl;
    ostrm << "Previous params: " << currentValues[0] << " " << currentValues[1]
          << std::endl
          << std::endl;
    if (transitionProbability < stepProbability) {
      currentValues = candidateValues;
      currentTarget = candidateTarget;
    }
    i++;
    stepFunction(candidateValues, currentValues, currentTemperature,
                 initialTemperature);
    currentTemperature = temperatureFunction(i, initialTemperature);
  }
  ostrm.close();
}

float SimulatedAnnealingOptimizer::getProbability(float currentTarget,
                                                  float candidateTarget,
                                                  float temperature) {
  if (candidateTarget > currentTarget) {
    return 1.0;
  }
  return exp((candidateTarget - currentTarget) / temperature);
}

} // namespace eda::hls::scheduler::optimizers