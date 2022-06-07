//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "hls/model/model.h"
#include "util/singleton.h"

using namespace eda::hls::library;
using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler::dse {

class DesignExplorer final : public Singleton<DesignExplorer> {

public:
  friend class Singleton<DesignExplorer>;

  void estimateIndicators(const Model &model,
      const std::map<std::string, Parameters> &params, Indicators &indicators) const {
  
    Graph *graph = model.main();
    const unsigned maxFrequency = 1000000;
    
    // Estimate the integral indicators.
    indicators.freq = maxFrequency;
    indicators.power = 0;
    indicators.area = 0;

    for (const auto *node : graph->nodes) {
      auto metaElement = Library::get().find(node->type);
      assert(metaElement && "MetaElement is not found");
      auto nodeParams = params.find(node->name);

      Indicators nodeIndicators;
      
      Parameters *tempParams = nullptr;
      if (nodeParams == params.end()) {
        // FIXME
        Constraint constr(1000, 500000);
        tempParams = new Parameters(metaElement->params);
        tempParams->add(Parameter("f", constr, constr.max)); // FIXME
        tempParams->add(Parameter("stages", Constraint(0, 10000), 10));  // FIXME
      }

      metaElement->estimate(
        nodeParams == params.end()
          ? *tempParams         // For inserted nodes 
          : nodeParams->second, // For existing nodes
        nodeIndicators);

      indicators.power += nodeIndicators.power;
      indicators.area += nodeIndicators.area;

      // Set the minimal frequency rate
      if (nodeIndicators.freq < indicators.freq) {
        indicators.freq = nodeIndicators.freq;
      }
    }
    indicators.perf = indicators.freq;
  }
};

} // namespace eda::hls::scheduler::dse

