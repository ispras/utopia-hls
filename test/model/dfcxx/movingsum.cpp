//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "movingsum/movingsum.h"

#include "gtest/gtest.h"

TEST(DFCxx, MovingSum_add_int_2_ASAP) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::ASAP), true);
}

TEST(DFCxx, MovingSum_add_int_2_Linear) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 2}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}

TEST(DFCxx, MovingSum_add_int_8_ASAP) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::ASAP), true);
}

TEST(DFCxx, MovingSum_add_int_8_Linear) {
  MovingSum kernel;
  DFLatencyConfig config = {
          {dfcxx::ADD_INT, 8}
  };
  EXPECT_EQ(kernel.compile(config, NULLDEVICE, dfcxx::Linear), true);
}