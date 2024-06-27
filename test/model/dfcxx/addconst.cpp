//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "addconst/addconst.h"

#include "gtest/gtest.h"

TEST(DFCxx, AddConstAddInt2Asap) {
  AddConst kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::ASAP), true);
}

TEST(DFCxx, AddConstAddInt2Linear) {
  AddConst kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}