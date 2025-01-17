//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "idct/idct.h"

#include "gtest/gtest.h"

static const DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(DFCxx, IdctAsap) {
  IDCT kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 1},
      {dfcxx::SUB_INT, 1},
      {dfcxx::MUL_INT, 3},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

// Issue #7 (https://github.com/ispras/utopia-hls/issues/7).

// TEST(DFCxx, IdctLinear) {
//   IDCT kernel;
//   DFLatencyConfig config = DFLatencyConfig(
//     {
//       {dfcxx::ADD_INT, 1},
//       {dfcxx::SUB_INT, 1},
//       {dfcxx::MUL_INT, 3},
//     },
//     {}
//   );
//   EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
// }
