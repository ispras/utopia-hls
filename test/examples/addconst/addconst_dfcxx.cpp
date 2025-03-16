//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "addconst/addconst.h"

#include "gtest/gtest.h"

static const dfcxx::DFOutputPaths nullDevicePath =
    {{dfcxx::OutputFormatID::SystemVerilog, NULLDEVICE}};

TEST(ExamplesAddConst, DFCXXAddInt2Asap) {
  AddConst kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::ASAP), true);
}

TEST(ExamplesAddConst, DFCXXAddInt2Linear) {
  AddConst kernel;
  dfcxx::DFLatencyConfig config = dfcxx::DFLatencyConfig(
    {
      {dfcxx::ADD_INT, 2}
    },
    {}
  );
  EXPECT_EQ(kernel.compile(config, nullDevicePath, dfcxx::Linear), true);
}
