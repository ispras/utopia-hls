//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "polynomial2/polynomial2.h"

#include "gtest/gtest.h"

static const DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(DFCxx, Polynomial2AddInt2MulInt3Asap) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
    {dfcxx::ADD_INT, 2},
    {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(DFCxx, Polynomial2AddInt2MulInt3Linear) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
    {dfcxx::ADD_INT, 2},
    {dfcxx::MUL_INT, 3}
  };
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}

TEST(DFCxx, Polynomial2AddInt8MulInt15Asap) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
    {dfcxx::ADD_INT, 8},
    {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(DFCxx, Polynomial2AddInt8MulInt15Linear) {
  Polynomial2 kernel;
  DFLatencyConfig config = {
    {dfcxx::ADD_INT, 8},
    {dfcxx::MUL_INT, 15}
  };
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
