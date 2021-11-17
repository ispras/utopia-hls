//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "gtest/gtest.h"

#include "hls/parser/dfc/dfc.h"

DFC_KERNEL(MyKernel) {
  DFC_INPUT(x, dfc::uint32);
  DFC_INPUT(y, dfc::uint32);
  DFC_OUTPUT(z, dfc::uint32);

  DFC_KERNEL_CTOR(MyKernel) {
    dfc::var<dfc::uint32> tmp = x;
    z = tmp + y;
  }
};

void dfcTest() {
  dfc::params args;
  MyKernel kernel(args);
}

TEST(DfcTest, SimpleTest) {
  dfcTest();
}
