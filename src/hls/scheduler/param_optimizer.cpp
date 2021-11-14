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
    //nodeParams.set("f", criteria.frequency.max);
    params.insert({ node->name, nodeParams });
  }

  std::ofstream ostrm("debug_file.txt", std::ios::out);
  ostrm << "Running optimization loop" << std::endl;
  unsigned cur_f = criteria.frequency.max;
  // Optimization loop.
  const unsigned N = 3;
  for (unsigned i = 0; i < N; i++) {
    // Update the values of the parameters.
    for (const auto *node : graph->nodes) {
      auto nodeParams = params.find(node->name);
      if (nodeParams == params.end())
        continue;

      /*for (auto param : nodeParams->second.params) {
        ostrm << param.second.name << ": " << param.second.value << std::endl;
        //param.set("f", param.second.value / 2);
      }*/
      nodeParams->second.set("f", cur_f);
    }

    // Balance flows and align times.
    LpSolver solver;
    solver.balance(model);

    indicators.latency = 100; // FIXME
    indicators.frequency = cur_f; // FIXME

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
    ostrm << "Iteration #" << i << std::endl;
    ostrm << std::endl << "Frequency: " << indicators.frequency << std::endl;
    ostrm << "Latency: " << indicators.latency << std::endl;
    ostrm << "Throughput: " << indicators.throughput << std::endl;
    ostrm << "Power: " << indicators.power << std::endl;
    ostrm << "Area: " << indicators.area << std::endl << std::endl;
    
    // Check the constraints.
    /*if (criteria.check(indicators)) {
      // Acceptable.
      break; // FIXME
    }*/

    cur_f = cur_f / 2;
    // Reset to the initial model state.
    model.undo();
  }

  ostrm.close();
  return params;
}

} // namespace eda::hls::scheduler
