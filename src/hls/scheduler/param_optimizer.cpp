//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"
#include "hls/model/model.h"
#include "hls/scheduler/dse/design_explorer.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/latency_solver.h"
#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace eda::hls::scheduler {

std::map<std::string, Parameters> ParametersOptimizer::optimize(
    const Criteria &criteria,
    Model &model,
    Indicators &indicators) const {
  srand(42);

  std::map<std::string, Parameters> params;

  // Get the main dataflow graph.
  Graph *graph = model.main();
  assert(graph != nullptr && "Main graph is not found");

  // Collect the parameters for all nodes.
  Parameters defaultParams;
  defaultParams.add(Parameter("f", criteria.frequency, criteria.frequency.max));

  auto min_value = criteria.frequency.min;
  auto max_value = criteria.frequency.max;

  std::vector<float> optimized_values;
  optimized_values.push_back(normalize(10000, min_value, max_value));
  for (const auto *node : graph->nodes) {
    auto metaElement = Library::get().find(node->type);
    Parameters nodeParams(metaElement->params);
    for(const auto& iter : metaElement->params.params) {
      optimized_values.push_back(normalize(iter.second.value, min_value, max_value));
    }
    params.insert({ node->name, nodeParams });
  }

  // Check if the task is solvable
  int cur_f = criteria.frequency.min;
  estimate(model, params, indicators, cur_f);
  model.undo();
  if (!criteria.check(indicators)) { // even if the frequency is minimal the params don't match constratints
      return params;
  }

  cur_f = criteria.frequency.max;
  int x2 = cur_f;
  estimate(model, params, indicators, cur_f);
  if (criteria.check(indicators)) { // the maximum frequency is the solution
      return params;
  }

  std::function<float(int, float)> temp_fun = [](int i, float temp) -> float {
    return temp / log(i + 1);
  };

  std::function<void(std::vector<float>&, const std::vector<float>&, float)>
    step_fun = [&](std::vector<float>& x, const std::vector<float>& prev, float temp) -> void {
      std::random_device rand_dev{};
      std::mt19937 gen{rand_dev()};
       std::normal_distribution<> distr{0, 1};

      for(int i = 0; i < x.size(); i++) {
        auto norm = normalize(prev[i], min_value, max_value);
        auto diff = abs(distr(gen));
        x[i] = norm + diff;
      }
    };

  std::function<float(const std::vector<float>&)> target_function = [&](const std::vector<float>& parameters) -> float {
    std::vector<float> denormalized_parameters;
    for(const auto& param : parameters) {
      denormalized_parameters.push_back(denormalize(param, min_value, max_value));
    }
    estimate(model, params, indicators, denormalized_parameters[0]);
    model.undo();
    return indicators.frequency;
  };

  std::function<float(const std::vector<float>&)>
      limitation_function = [&](const std::vector<float>& parameters) -> float {
        float tmp = parameters[0];
        tmp = denormalize(tmp, min_value, max_value);
        estimate(model, params, indicators, tmp);
        model.undo();
        return indicators.area;
      };

  float init_temp = 10000000000.0;
  float end_temp = 1.0;
  float limit = 10000;

  eda::hls::scheduler::optimizers::simulated_annealing_optimizer test(init_temp, end_temp, limit, target_function,
                                                                      limitation_function, step_fun, temp_fun);
  test.optimize(optimized_values);

  auto res_freq = target_function(optimized_values);
  auto limitation = limitation_function(optimized_values);

  std::ofstream ostrm("real.txt");

  ostrm << std::endl << "After optimization" << std::endl;
  ostrm << "Frequency: " << indicators.frequency << std::endl;
  ostrm << "Throughput: " << indicators.throughput << std::endl;
  ostrm << "Latency: " << indicators.latency << std::endl;
  ostrm << "Power: " << indicators.power << std::endl;
  ostrm << "Area: " << indicators.area << std::endl;

  ostrm << "Target function: " << res_freq << std::endl;
  ostrm << "Limitation: " << limitation << std::endl;

  ostrm.close();
  return params;
}

void ParametersOptimizer::updateFrequency(Model& model,
    std::map<std::string, Parameters>& params,
    const unsigned frequency) const {
  Graph *graph = model.main();
  for (const auto *node : graph->nodes) {
    auto nodeParams = params.find(node->name);
    if (nodeParams == params.end()) {
      continue;
    }
    nodeParams->second.set("f", frequency);
  }
}


void ParametersOptimizer::estimate(Model& model,
    std::map<std::string, Parameters>& params,
    Indicators& indicators, unsigned frequency) const {
  // Update the values of the parameters.
  updateFrequency(model, params, frequency);
  // Balance flows and align times.
  LatencyLpSolver::get().balance(model);
  indicators.latency = LatencyLpSolver::get().getGraphLatency();
  // Estimate overall design indicators
  dse::DesignExplorer::get().estimateIndicators(model, params, indicators);

}

double ParametersOptimizer::normalize(double value, double min, double max) const {
  return (value - min) / (max - min);
}

double ParametersOptimizer::denormalize(double value, double min, double max) const {
  return value * (max - min) + min;
}

} // namespace eda::hls::scheduler
