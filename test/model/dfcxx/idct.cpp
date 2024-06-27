//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "idct/idct.h"

#include "gtest/gtest.h"

TEST(DFCxx, IDCT_add_int_1_mul_int_3_sub_int_1_ASAP) {
  IDCT kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 1},
          {dfcxx::MUL_INT, 3},
          {dfcxx::SUB_INT, 1}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::ASAP), true);
}

// Issue #7 (https://github.com/ispras/utopia-hls/issues/7).

// TEST(DFCxx, IDCT_add_int_1_mul_int_3_sub_int_1_Linear) {
//   IDCT kernel;
//   DFLatencyConfig config = {
//           {dfcxx::ADD_INT, 1},
//           {dfcxx::MUL_INT, 3},
//           {dfcxx::SUB_INT, 1}
//   };
//   EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
// }
