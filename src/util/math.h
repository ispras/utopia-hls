//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>

namespace eda::utils {

inline unsigned factorial(unsigned n) {
  unsigned result[] = {
    /*  0 */ 1,
    /*  1 */ 1,
    /*  2 */ 2,
    /*  3 */ 6,
    /*  4 */ 24,
    /*  5 */ 120,
    /*  6 */ 720,
    /*  7 */ 5040,
    /*  8 */ 40320,
    /*  9 */ 362880,
    /* 10 */ 3628800,
    /* 11 */ 39916800,
    /* 12 */ 479001600
  };

  assert(n < 13);
  return result[n];
}

} // namespace eda::utils
