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

#include <cassert>

namespace eda::hls::scheduler {

std::map<std::string, Parameters> ParametersOptimizer::optimize(
    const Model &model,
    const Criteria &criteria,
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

  // Optimization loop.
  while (true) {
    // Update the values of the parameters.
    for (const auto *node : graph->nodes) {
      auto i = params.find(node->name);
      for (auto param : i->second.params) {
        param.second.value++; // FIXME
      }
    }

    // Balance flows and align times.
    // TODO:
    indicators.latency = 100; // FIXME
    indicators.frequency = 100000; // FIXME

    // Estimate the integral indicators.
    indicators.throughput = indicators.frequency;
    indicators.power = 0;
    indicators.area = 0;

    for (const auto *node : graph->nodes) {
      auto metaElement = Library::get().find(node->type);
      auto i = params.find(node->name);

      Indicators nodeIndicators;
      metaElement->estimate(i->second, nodeIndicators);

      indicators.power += nodeIndicators.power;
      indicators.area += nodeIndicators.area;
    }

    // Check the constraints.
    if (criteria.check(indicators)) {
      // Acceptable.
      break; // FIXME
    } else {
      // Unacceptable.
      break; // FIXME
    }
  }

  return params;
}

} // namespace eda::hls::scheduler
