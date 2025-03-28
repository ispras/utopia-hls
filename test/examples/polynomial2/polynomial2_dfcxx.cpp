//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "polynomial2/polynomial2.h"

#include "gtest/gtest.h"

static const dfcxx::DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(ExamplesPolynomial2, DFCXXAddInt2MulInt3Asap) {
  Polynomial2 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesPolynomial2, DFCXXAddInt2MulInt3Linear) {
  Polynomial2 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2},
      {dfcxx::MUL_INT, 3}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}

TEST(ExamplesPolynomial2, DFCXXAddInt8MulInt15Asap) {
  Polynomial2 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 8},
      {dfcxx::MUL_INT, 15}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesPolynomial2, DFCXXAddInt8MulInt15Linear) {
  Polynomial2 kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 8},
      {dfcxx::MUL_INT, 15}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
