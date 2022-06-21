//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <limits>

namespace eda::hls::model {

class Constraint final {
public:
  Constraint(unsigned min, unsigned max):
    min(min), max(max) {}

  Constraint():
    min(0), max(std::numeric_limits<unsigned>::max()) {}

  Constraint(const Constraint &) = default;

  bool check(unsigned value) const {
    return min <= value && value <= max;
  }

  unsigned getMin() const { return min; }
  unsigned getMax() const { return max; }

private:
  const unsigned min;
  const unsigned max;
};

} // namespace eda::hls::model
