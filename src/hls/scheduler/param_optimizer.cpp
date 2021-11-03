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
  defaultParams.add(Parameter("f", Constraint(1, 1000), 100)); // FIXME

  for (const auto *node : graph->nodes) {
    auto metaElement = Library::get().find(node->type);
    Parameters nodeParams(node->name, metaElement->params);
    params.insert({ node->name, nodeParams });
  }

  // Optimization loop.
  const unsigned N = 2;
  for (unsigned i = 0; i < N; i++) {
    // Update the values of the parameters.
    for (const auto *node : graph->nodes) {
      auto nodeParams = params.find(node->name);
      if (nodeParams == params.end())
        continue;

      for (auto param : nodeParams->second.params) {
        param.second.value /= 2; // FIXME
      }
    }

    // Balance flows and align times.
    LpSolver solver;
    solver.balance(model);

    indicators.latency = 100; // FIXME
    indicators.frequency = 100000; // FIXME

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

    // Check the constraints.
    if (criteria.check(indicators)) {
      // Acceptable.
      break; // FIXME
    }

    // Reset to the initial model state.
    model.undo();
  }

  return params;
}

} // namespace eda::hls::scheduler
