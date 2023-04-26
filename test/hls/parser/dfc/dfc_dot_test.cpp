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

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

DFC_KERNEL(DotProduct) {
  static const std::size_t N = 8;

  DFC_KERNEL_CTOR(DotProduct) {
    std::array<dfc::input<dfc::uint32>, N> x, y;
    dfc::stream<dfc::uint32> z;

    z = x[0] * y[0];
    for (std::size_t i = 1; i < N; i++) {
      z += x[i] * y[i];
    }
  }
};

void dfcDotTest(const std::string &outSubPath) {
  dfc::params args;
  DotProduct kernel(args);

  std::shared_ptr<eda::hls::model::Model> model =
    eda::hls::parser::dfc::Builder::get().create("DotModel");
  std::cout << *model << std::endl;

  const fs::path homePath = std::string(getenv("UTOPIA_HLS_HOME"));
  const fs::path fsOutPath = homePath / outSubPath;
  fs::create_directories(fsOutPath);

  const std::string outDotFileName = "dfc_dot_test.dot";

  std::ofstream output(fsOutPath / outDotFileName);

  eda::hls::model::printDot(output, *model);
  output.close();
}

TEST(DfcTest, DfcDotTest) {
  dfcDotTest("output/test/dfc_dot_test/");
}
