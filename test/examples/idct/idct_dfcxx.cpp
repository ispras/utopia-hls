//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "idct/idct.h"

#include "gtest/gtest.h"

static const dfcxx::DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(ExamplesIDCT, DFCXXAsap) {
  IDCT kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 1},
      {dfcxx::SUB_INT, 1},
      {dfcxx::MUL_INT, 3},
      {dfcxx::GREATER_INT, 3},
      {dfcxx::GREATEREQ_INT, 3},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

// Issue #7 (https://github.com/ispras/utopia-hls/issues/7).

// TEST(ExamplesIDCT, DFCXXLinear) {
//   IDCT kernel;
//   dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
//     {
//       {dfcxx::ADD_INT, 1},
//       {dfcxx::SUB_INT, 1},
//       {dfcxx::MUL_INT, 3},
//       {dfcxx::GREATEREQ_INT, 3},
//       {dfcxx::GREATEREQ_INT, 3},
//     },
//     {}
//   );
//   EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
// }
