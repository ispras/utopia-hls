//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

int foo(int x) {
  return x >= 0 ? x : -x;
}

TEST(FooTest, PositiveValues) {
  EXPECT_EQ(foo(1), 1);
  EXPECT_EQ(foo(2), 2);
}

TEST(FooTest, NegativeValues) {
  EXPECT_EQ(foo(-1), 1);
  EXPECT_EQ(foo(-2), 2);
}

TEST(FooTest, ZeroValue) {
  EXPECT_EQ(foo(0), 0);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

