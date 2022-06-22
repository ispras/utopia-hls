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

template <typename T>
class Constraint final {
public:
  Constraint(T min, T max):
    min(min), max(max) {}

  Constraint():
    min(std::numeric_limits<T>::min()),
    max(std::numeric_limits<T>::max()) {}

  Constraint(const Constraint<T> &) = default;

  bool check(T value) const {
    return min <= value && value <= max;
  }

  T getMin() const { return min; }
  T getMax() const { return max; }

private:
  const T min;
  const T max;
};

} // namespace eda::hls::model
