//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/param_optimizer.h"

namespace eda::hls::scheduler {

std::map<std::string, Parameters> ParametersOptimizer::optimize(
    const Model &model,
    const Criteria &criteria,
    Indicators &indicators) const {
  std::map<std::string, Parameters> params;

  // TODO: Iterate over the model's nodes.
  // model.nodes

  // TODO: For each node:

    // TODO: Get the node's parameters.
    // Library::get().find(node.type.name)

    // TODO: Set the parameters' values.
    // params.set(name, value)

    // TODO: Insert the buffers to align times.

    // TODO: Estimate the indicators.
    // Library::get().estimate(params, indicators)

  // TODO: Estimate the integral indicators.

  // TODO: Check the constraints.

  // TODO: Search for better solution.

  return params;
}

} // namespace eda::hls::scheduler
