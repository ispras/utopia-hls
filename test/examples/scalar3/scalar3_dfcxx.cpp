//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "scalar3/scalar3.h"

#include "gtest/gtest.h"

static const dfcxx::DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(ExamplesScalar3, DFCXXAddInt2MulInt3Asap) {
  Scalar3 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesScalar3, DFCXXAddInt2MulInt3Linear) {
  Scalar3 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
