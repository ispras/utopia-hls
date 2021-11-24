//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/parser/dfc/dfc.h"
#include "hls/parser/dfc/internal/builder.h"

#include "gtest/gtest.h"

#include <iostream>

DFC_KERNEL(MyKernel) {
  static constexpr std::size_t N = 10;

  DFC_KERNEL_CTOR(MyKernel) {
    std::array<dfc::input<dfc::uint32>, N> x;
    std::array<dfc::input<dfc::uint32>, N> y;
    dfc::output<dfc::uint32> z;
  
    kernel(x, y, z);
  }

  void add_kernel(dfc::input<dfc::uint32> x,
                  dfc::input<dfc::uint32> y,
                  dfc::output<dfc::uint32> z) {
    z = x + y;
  }

  void mul_kernel(dfc::input<dfc::uint32> x,
                  dfc::input<dfc::uint32> y,
                  dfc::output<dfc::uint32> z) {
    z = x * y;
  }

  void kernel(std::array<dfc::input<dfc::uint32>, N> x,
              std::array<dfc::input<dfc::uint32>, N> y,
              dfc::output<dfc::uint32> z) {
    std::array<dfc::stream<dfc::uint32>, N> m;
    for (std::size_t i = 0; i < N; i++) {
      mul_kernel(x[i], y[i], m[i]);
    }

    dfc::stream<dfc::uint32> acc = m[0];
    for (std::size_t i = 1; i < N; i++) {
      add_kernel(m[i], acc, acc);
    }
  }
};

void dfcTest() {
  dfc::params args;
  MyKernel kernel(args);

  std::shared_ptr<eda::hls::model::Model> model = eda::hls::parser::dfc::Builder::get().create("MyModel");
  std::cout << model << std::endl;
}

TEST(DfcTest, SimpleTest) {
  dfcTest();
}
