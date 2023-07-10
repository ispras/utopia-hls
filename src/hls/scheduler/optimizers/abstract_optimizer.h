//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

// C++ STL
#include <vector>

namespace eda::hls::scheduler::optimizers {

class AbstractOptimizer {
public:
  AbstractOptimizer();
  virtual void optimize(std::vector<float> &parameterValues) = 0;
};

} // namespace eda::hls::scheduler::optimizers