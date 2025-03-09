//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "muxmul/muxmul.h"

#include "gtest/gtest.h"

static const DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(ExamplesMuxMul, DFCXXAddInt2MulInt3Asap) {
  MuxMul kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesMuxMul, DFCXXAddInt2MulInt3Linear) {
  MuxMul kernel;
  DFLatencyConfig config = DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
