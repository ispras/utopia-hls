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

int dfcTest() {
  dfc::sint8 scalar;
  dfc::vector<dfc::uint16, 128> vector;
  dfc::tuple<dfc::uint16, dfc::complex<dfc::sint8>> tuple;
  dfc::tensor<dfc::uint8, 2, 3> tensor;

  std::cout << scalar.name() << ": " << scalar.size() << std::endl;
  std::cout << vector.name() << ": " << vector.size() << std::endl;
  std::cout << tuple.name()  << ": " << tuple.size()  << std::endl;
  std::cout << tensor.name() << ": " << tensor.size() << std::endl;
}

TEST(DfcTest, SimpleTest) {
  dfcTest();
}
