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
  Parameters defaultParams("<default>");
  defaultParams.add(Parameter("f", criteria.frequency, criteria.frequency.max)); // FIXME

  for (const auto *node : graph->nodes) {
    auto metaElement = Library::get().find(node->type);
    Parameters nodeParams(node->name, metaElement->params);
    params.insert({ node->name, nodeParams });
  }

  int y1, y2;

  // Check if the task is solvable
  int cur_f = criteria.frequency.min;
  count_params(model, params, indicators, cur_f, defaultParams);
  if (!criteria.check(indicators)) { // even if the frequency is minimal the params don't match constratints
      return params;
  }

  cur_f = criteria.frequency.max;
  int x2 = cur_f;
  count_params(model, params, indicators, cur_f, defaultParams);
  if (criteria.check(indicators)) { // the maximum frequency is the solution
      return params;
  }
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
  }

  return params;
}

void ParametersOptimizer::count_params(Model& model,
                                      std::map<std::string, Parameters>& params,
                                      Indicators& indicators, unsigned f,
                                      Parameters& defaultParams) const {
  Graph *graph = model.main();
  // Update the values of the parameters.
  for (const auto *node : graph->nodes) {
    auto nodeParams = params.find(node->name);
    if (nodeParams == params.end()) {
      continue;
    }
        
    nodeParams->second.set("f", f);
  }

  // Balance flows and align times.
  LpSolver::get().balance(model);
    
  indicators.latency = LpSolver::get().getGraphLatency();
  indicators.frequency = f;

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

} // namespace eda::hls::scheduler
