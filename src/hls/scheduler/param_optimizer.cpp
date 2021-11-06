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
#include <iostream>
#include <fstream>
#include <string>

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
    const MetaElement &metaElement = Library::get().find(node->type);
    Parameters nodeParams(node->name, metaElement.params);
    params.insert({ node->name, nodeParams });
  }

  int iter_counter = 0;
  std::ofstream ostrm("debug_file.txt", std::ios::out);
  ostrm << "Running optimization loop" << std::endl;
  // Optimization loop.
  while (true) {
    iter_counter++;
    // Update the values of the parameters.
    for (const auto *node : graph->nodes) {
      auto i = params.find(node->name);
      for (auto param : i->second.params) {
        param.second.value += 100; // FIXME
      }
    }

    // Balance flows and align times.
    // TODO:
    indicators.latency = 100; // FIXME
    indicators.frequency = 100000 * iter_counter; // FIXME

    // Estimate the integral indicators.
    indicators.throughput = indicators.frequency;
    indicators.power = 0;
    indicators.area = 0;

    for (const auto *node : graph->nodes) {
      auto i = params.find(node->name);
      Indicators nodeIndicators;
      Library::get().estimate(i->second, nodeIndicators);

      //ostrm << i->second.value("f") << std::endl;
      ostrm << nodeIndicators.frequency << std::endl;

      indicators.power += nodeIndicators.power;
      indicators.area += nodeIndicators.area;
    }
    ostrm << std::endl << "Frequency: " << indicators.frequency << std::endl;
    ostrm << "Latency: " << indicators.latency << std::endl;
    ostrm << "Throughput: " << indicators.throughput << std::endl;
    ostrm << "Power: " << indicators.power << std::endl;
    ostrm << "Area: " << indicators.area << std::endl << std::endl;
    
    // Check the constraints.
    if (criteria.check(indicators)) {
      // Acceptable.
      ostrm << "Solution found" << std::endl;
      break;
    } else {
      // Unacceptable.
      if(iter_counter > 10) {
        ostrm << "Solution has not been found" << std::endl;
        break;
      }
    }
    //break;
  }

  ostrm << "Number of iterations: " << iter_counter << std::endl;
  ostrm.close();
  return params;
}

} // namespace eda::hls::scheduler
