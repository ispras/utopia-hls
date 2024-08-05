//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "polynomial2/polynomial2.h"

#include "gtest/gtest.h"

static const Polynomial2 kernel;

static const DFLatencyConfig config =
    {{dfcxx::ADD_INT, 2}, {dfcxx::MUL_INT, 3}};

TEST(DFCxxOutputFormats, FIRRTL) {
  Polynomial2 kernel;
  DFOutputPaths paths = {
    {dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}
  };
  EXPECT_EQ(kernel.compile(config, paths, dfcxx::ASAP), true);
}
