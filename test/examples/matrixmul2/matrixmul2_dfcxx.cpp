//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "matrixmul2/matrixmul2.h"

#include "gtest/gtest.h"

static const dfcxx::DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(ExamplesMatrixMul2, DFCXXAddInt2MulInt3Asap) {
  MatrixMul2 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesMatrixMul2, DFCXXAddInt2MulInt3Linear) {
  MatrixMul2 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3},
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
