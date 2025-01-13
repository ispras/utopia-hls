//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "movingsum/movingsum.h"

#include "gtest/gtest.h"

static const DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(DFCxx, MovingSumAddInt2Asap) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(DFCxx, MovingSumAddInt2Linear) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}

TEST(DFCxx, MovingSumAddInt8Asap) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 8},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(DFCxx, MovingSumAddInt8Linear) {
  MovingSum kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 8},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
