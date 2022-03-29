//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"
#include "hls/model/model.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/solver.h"
#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

namespace eda::hls::scheduler {

std::map<std::string, Parameters> ParametersOptimizer::optimize(
    const Criteria &criteria,
    Model &model,
    Indicators &indicators) const {
  std::map<std::string, Parameters> params;

  // Get the main dataflow graph.
  Graph *graph = model.main();
  assert(graph != nullptr && "Main graph is not found");

  // Collect the parameters for all nodes.
  Parameters defaultParams("<default>");
  defaultParams.add(Parameter("f", criteria.frequency, criteria.frequency.max));

  auto min_value = criteria.frequency.min;
  auto max_value = criteria.frequency.max;

  std::ofstream ostrm("real.txt");
  std::vector<float> optimized_values;
  optimized_values.push_back(normalize(10000, min_value, max_value));
  for (const auto *node : graph->nodes) {
    auto metaElement = Library::get().find(node->type);
    Parameters nodeParams(node->name, metaElement->params);
    for(const auto& iter : metaElement->params.params) {
      optimized_values.push_back(normalize(iter.second.value, min_value, max_value));
    }
    params.insert({ node->name, nodeParams });
  }

  // Check if the task is solvable
  int cur_f = criteria.frequency.min;
  count_params(model, params, indicators, cur_f, defaultParams);
  model.undo();
  if (!criteria.check(indicators)) { // even if the frequency is minimal the params don't match constratints
      return params;
  }

  cur_f = criteria.frequency.max;
  count_params(model, params, indicators, cur_f, defaultParams);
  model.undo();
  if (criteria.check(indicators)) { // the maximum frequency is the solution
      return params;
  }

  std::function<float(int, float)> temp_fun = [](int i, float temp) -> float {
    return temp / log(i + 1);
  };

  std::function<void(std::vector<float>&, const std::vector<float>&, float)>
    step_fun = [&](std::vector<float>& x, const std::vector<float>& prev, float temp) -> void {
      for(int i = 0; i < x.size(); i++) {
        auto norm = normalize(prev[i], min_value, max_value);
        auto diff = exp((-0.5 * prev[i] * prev[i])) / (std::sqrt(M_PI * 2));
        x[i] = prev[i] + diff;
      }
    };

  std::function<float(const std::vector<float>&)> target_function = [&](const std::vector<float>& parameters) -> float {
    ostrm << "Current parameters before norm: " << std::endl;
    ostrm << parameters[0] << " " << parameters[1] << std::endl;
    std::vector<float> denormalized_parameters;
    for(const auto& param : parameters) {
      denormalized_parameters.push_back(denormalize(param, min_value, max_value));
    }
    count_params(model, params, indicators, denormalized_parameters[0], defaultParams);
    model.undo();
    ostrm << "Current parameters: " << std::endl;
    ostrm << denormalized_parameters[0] << " " << denormalized_parameters[1] << std::endl;
    return indicators.frequency;
  };

  std::function<float(const std::vector<float>&)>
      limitation_function = [&](const std::vector<float>& parameters) -> float {
        float tmp = parameters[0];
        tmp = denormalize(tmp, min_value, max_value);
        ostrm << "Counting area for freq: " << tmp << std::endl;
        count_params(model, params, indicators, tmp, defaultParams);
        model.undo();
        ostrm << "Area: " << indicators.area << std::endl;
        return indicators.area;
      };

  float init_temp = 10000000000.0;
  float end_temp = 1.0;
  float limit = 10000;

  eda::hls::scheduler::optimizers::simulated_annealing_optimizer test(init_temp, end_temp, limit, target_function,
                                                                      limitation_function, step_fun, temp_fun);
  test.optimize(optimized_values);

  ostrm << std::endl << "After optimization" << std::endl;
  ostrm << "Frequency: " << indicators.frequency << std::endl;
  ostrm << "Throughput: " << indicators.throughput << std::endl;
  ostrm << "Latency: " << indicators.latency << std::endl;
  ostrm << "Power: " << indicators.power << std::endl;
  ostrm << "Area: " << indicators.area << std::endl;

  ostrm.close();
  /*int y1, y2;
  int x2 = cur_f;
  y2 = indicators.area;
  int x1 = criteria.frequency.max - (criteria.frequency.max - criteria.frequency.min) / 10;
  count_params(model, params, indicators, x1, defaultParams);
  y1 = indicators.area;
  float k = float(y1 - y2) / float(x1 - x2);
  float b = float(y1 - x1 * k);
  cur_f = (criteria.area.max - b) / k;
  count_params(model, params, indicators, cur_f, defaultParams);
  int sign;
  if(indicators.area > criteria.area.max) {
    sign = -1;
  } else {
    sign = 1;
  }
  int grid = (criteria.frequency.max - criteria.frequency.min) / 100;

  // Optimization loop.
  const unsigned N = 5;
  for (unsigned i = 0; i < N; i++) {

    cur_f += sign * grid;
    count_params(model, params, indicators, cur_f, defaultParams);
    // Check the constraints.
    if (criteria.check(indicators)) {
      break;
    }
    // Reset to the initial model state.
    model.undo();
  }*/

  return params;
}

void ParametersOptimizer::count_params(Model& model,
                                      std::map<std::string, Parameters>& params,
                                      Indicators& indicators, unsigned frequency,
                                      Parameters& defaultParams) const {
  Graph *graph = model.main();
  // Update the values of the parameters.
  for (const auto *node : graph->nodes) {
    auto nodeParams = params.find(node->name);
    if (nodeParams == params.end()) {
      continue;
    }   
    nodeParams->second.set("f", frequency);
  }

  // Balance flows and align times.
  LpSolver::get().balance(model);
    
  indicators.latency = LpSolver::get().getGraphLatency();
  indicators.frequency = frequency;

  // Estimate the integral indicators.
  indicators.throughput = indicators.frequency;
  indicators.power = 0;
  indicators.area = 0;

  for (const auto *node : graph->nodes) {
    auto metaElement = Library::get().find(node->type);
    assert(metaElement && "MetaElement is not found");
    auto nodeParams = params.find(node->name);

    Indicators nodeIndicators;

    metaElement->estimate(
      nodeParams == params.end()
        ? defaultParams       // For inserted nodes 
        : nodeParams->second, // For existing nodes
      nodeIndicators);

    indicators.power += nodeIndicators.power;
    indicators.area += nodeIndicators.area;

  }
}

double ParametersOptimizer::normalize(double value, double min, double max) const {
  return (value - min) / (max - min);
}

double ParametersOptimizer::denormalize(double value, double min, double max) const {
  return value * (max - min) + min;
}

} // namespace eda::hls::scheduler
