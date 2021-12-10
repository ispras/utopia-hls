//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/dfc/dfc.h"
#include "hls/parser/dfc/internal/builder.h"

#include "gtest/gtest.h"

#include <fstream>
#include <iostream>

DFC_KERNEL(DotProduct) {
  static constexpr std::size_t N = 4;

  DFC_KERNEL_CTOR(DotProduct) {
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
    std::array<dfc::stream<dfc::uint32>, N> a;

    for (std::size_t i = 0; i < N; i++) {
      mul_kernel(x[i], y[i], m[i].to_output());
    }

    a[0] = m[0];
    for (std::size_t i = 1; i < N; i++) {
      add_kernel(a[i-1].to_input(), m[i].to_input(), a[i].to_output());
    }

    z = a[N-1];
  }
};

void dfcDotTest() {
  dfc::params args;
  DotProduct kernel(args);

  std::shared_ptr<eda::hls::model::Model> model =
    eda::hls::parser::dfc::Builder::get().create("DotModel");
  std::cout << *model << std::endl;

  std::ofstream output("dfc_dot_test.dot");
  eda::hls::model::printDot(output, *model);
  output.close();
}

TEST(DfcTest, DfcDotTest) {
  dfcDotTest();
}