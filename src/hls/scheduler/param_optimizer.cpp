//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"
#include "hls/mapper/mapper.h"
#include "hls/model/model.h"
#include "hls/scheduler/latency_solver.h"
#include "hls/scheduler/optimizers/simulated_annealing_optimizer.h"
#include "hls/scheduler/param_optimizer.h"

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
  std::ofstream ostrm("real.txt");

  std::map<std::string, Parameters> params;
  // Get the main dataflow graph.
  Graph *graph = model.main();
  assert(graph != nullptr && "Main graph is not found");

  std::vector<float> optimized_values, min_values, max_values;
  for (const auto *node : graph->nodes) {
    auto metaElement = node->map; 
    Parameters nodeParams(metaElement->params);
    for(const auto& iter : metaElement->params.getAll()) {
      optimized_values.push_back(0.5);
      min_values.push_back(iter.second.getMin());
      max_values.push_back(iter.second.getMax());
    }
    params.insert({ node->name, nodeParams });
  }

  std::function<float(int, float)> temp_fun = [](int i, float temp) -> float {
    return temp / log(i + 1);
  };

  std::function<void(std::vector<float>&, const std::vector<float>&, float)>
    step_fun = [&](std::vector<float>& x, const std::vector<float>& prev, float temp) -> void {
      std::random_device rand_dev{};
      std::mt19937 gen{rand_dev()};
      std::normal_distribution<> distr{0, 0.2};

      for(std::size_t i = 0; i < x.size(); i++) {
        auto norm = normalize(prev[i], min_values[i], max_values[i]);
        auto diff = distr(gen);
        x[i] = norm + diff;
        if(x[i] > 1) {
          x[i] = 1;
        }
        if (x[i] < 0) {
          x[i] = 0;
        }
      }
    };

  std::function<float(const std::vector<float>&)> target_function = [&](const std::vector<float>& parameters) -> float {
    std::vector<float> denormalized_parameters;
    std::size_t index = 0;
    for(const auto& param : parameters) {
      denormalized_parameters.push_back(denormalize(param, min_values[index], max_values[index]));
      index++;
    }

    estimate(model, params, indicators, denormalized_parameters);
    model.undo();
    return indicators.freq();
  };

  std::function<float(const std::vector<float>&)>
      limitation_function = [&](const std::vector<float>& parameters) -> float {
        std::vector<float> denormalized_parameters;
        std::size_t index = 2;
        for(const auto& param : parameters) {
          denormalized_parameters.push_back(denormalize(param, min_values[index], max_values[index]));
          index++;
        }
        estimate(model, params, indicators, denormalized_parameters);
        model.undo();
        return indicators.area;
      };

  float init_temp = 10000000000.0;
  float end_temp = 1.0;
  float limit = criteria.area.getMax();

  eda::hls::scheduler::optimizers::simulated_annealing_optimizer test(init_temp, end_temp, limit, target_function,
                                                                      limitation_function, step_fun, temp_fun);
  test.optimize(optimized_values);

  auto res_freq = target_function(optimized_values);
  auto limitation = limitation_function(optimized_values);

  ostrm << std::endl << "After optimization" << std::endl;
  ostrm << "Parameters values" << std::endl;
  for(auto val : optimized_values) {
    ostrm << val << " ";
  }
  ostrm << std::endl;

  ostrm << "Freq: " << indicators.freq() << std::endl;
  ostrm << "Perf: " << indicators.perf() << std::endl;
  ostrm << "Ticks: " << indicators.ticks << std::endl;
  ostrm << "Power: " << indicators.power << std::endl;
  ostrm << "Area: " << indicators.area << std::endl;

  ostrm << "Target function: " << res_freq << std::endl;
  ostrm << "Limitation: " << limitation << std::endl;

  ostrm.close();
  return params;
}

void ParametersOptimizer::estimate(Model& model,
    std::map<std::string, Parameters>& params,
    Indicators& indicators,
    const std::vector<float>& optimized_params) const {
  std::ofstream ostrm("estimation.txt");
  // Update the values of the parameters & apply to nodes.
  Graph *graph = model.main();
  std::size_t index = 0;
  for (auto *node : graph->nodes) {
    auto nodeParams = params.find(node->name);
    if (nodeParams == params.end()) {
      continue;
    }
    if (index % 2 == 0) {
      ostrm << "Set frequency: " << optimized_params[index] << std::endl;
      nodeParams->second.setValue("f", optimized_params[index]);
    } else {
      ostrm << "Set stages " << optimized_params[index] << std::endl;
      nodeParams->second.setValue("stages", optimized_params[index]);
    }
    mapper::Mapper::get().apply(*node, nodeParams->second);
    index++;
  }
  ostrm.close();
  
  // Balance flows and align times.
  LatencyLpSolver::get().balance(model);
  
  // Estimate overall design indicators
  mapper::Mapper::get().estimate(model);
  indicators = model.ind;
}

double ParametersOptimizer::normalize(double value, double min, double max) const {
  return (value - min) / (max - min);
}

double ParametersOptimizer::denormalize(double value, double min, double max) const {
  return value * (max - min) + min;
}

} // namespace eda::hls::scheduler
