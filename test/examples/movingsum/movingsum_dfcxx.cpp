//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "movingsum/movingsum.h"

#include "gtest/gtest.h"

static const DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(ExamplesMovingSum, DFCXXAddInt2Asap) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesMovingSum, DFCXXAddInt2Linear) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}

TEST(ExamplesMovingSum, DFCXXAddInt8Asap) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 8},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesMovingSum, DFCXXAddInt8Linear) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 8},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
