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

#include <cassert>
#include <iostream>
#include <fstream>
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
  for (const auto *node : graph->nodes) {
    auto metaElement = Library::get().find(node->type);
    Parameters nodeParams(node->name, metaElement->params);
    params.insert({ node->name, nodeParams });
  }

  std::ofstream ostrm("debug_file.txt", std::ios::out);

  int y1, y2;

  // Check if the task is solvable
  int cur_f = criteria.frequency.min;
  estimate(model, params, indicators, cur_f);
  model.undo();
  if (!criteria.check(indicators)) { // even if the frequency is minimal the params don't match constratints
      ostrm << "There is no solution" << std::endl;
      ostrm << "Frequency: " << indicators.frequency << std::endl;
      ostrm << "Latency: " << indicators.latency << std::endl;
      ostrm << "Throughput: " << indicators.throughput << std::endl;
      ostrm << "Power: " << indicators.power << std::endl;
      ostrm << "Area: " << indicators.area << std::endl;
      ostrm.close();
      return params;
  }

  cur_f = criteria.frequency.max;
  int x2 = cur_f;
  estimate(model, params, indicators, cur_f);
  if (criteria.check(indicators)) { // the maximum frequency is the solution
      ostrm << "Maximum frequency is the solution" << std::endl;
      ostrm << "Frequency: " << indicators.frequency << std::endl;
      ostrm << "Latency: " << indicators.latency << std::endl;
      ostrm << "Throughput: " << indicators.throughput << std::endl;
      ostrm << "Power: " << indicators.power << std::endl;
      ostrm << "Area: " << indicators.area << std::endl;
      ostrm.close();
      return params;
  }
  model.undo();
  y2 = indicators.area;
  ostrm << "First step: " << x2 << " " << y2 << std::endl;

  int x1 = criteria.frequency.max - (criteria.frequency.max - criteria.frequency.min) / 10;
  estimate(model, params, indicators, x1);
  model.undo();
  y1 = indicators.area;
  ostrm << "Second step: " << x1 << " " << y1 << std::endl;

  float k = float(y1 - y2) / float(x1 - x2);
  float b = float(y1 - x1 * k);

  ostrm << "k: " << k << " b: " << b << std::endl;
  cur_f = (criteria.area.max - b) / k;
  ostrm << "estimate: " << cur_f << std::endl;
  estimate(model, params, indicators, cur_f);
  model.undo();
  ostrm << "area: " << indicators.area << std::endl;
  
  int sign;
  if(indicators.area > criteria.area.max) {
    sign = -1;
  } else {
    sign = 1;
  }
  int grid = (criteria.frequency.max - criteria.frequency.min) / 100;

  std::ofstream csv("exp.csv", std::ios::out);
  ostrm << "Running optimization loop" << std::endl;
  csv << "Frequency,Throughput,Latency,Power,Area" << std::endl;

  // Optimization loop.
  const unsigned N = 5;
  for (unsigned i = 0; i < N; i++) {

    cur_f += sign * grid;

    ostrm << "Cur f: " << cur_f << std::endl;
    estimate(model, params, indicators, cur_f);
    ostrm << "Cur a: " << indicators.area << std::endl;
    csv << indicators.frequency << "," << indicators.throughput << "," << indicators.latency << "," << indicators.power << "," << indicators.area << std::endl;
    
    // Check the constraints.
    if (criteria.check(indicators)) {
      ostrm << "Break because solution is found. Number of iterations is: " << i;
      break;
    }

    // Reset to the initial model state.
    model.undo();
  }

  ostrm.close();
  csv.close();
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

} // namespace eda::hls::scheduler
