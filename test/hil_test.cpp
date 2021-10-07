//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>

#include "gtest/gtest.h"

#include "hls/model/model.h"
#include "hls/library/library.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::library;
using namespace eda::hls::model;
using namespace eda::hls::parser::hil;

int hil_test(const std::string &filename) {
  std::cout << *parse(filename);
  return 0;
}

int hil_test_nodetypes(const std::string &filename) {
  return (parse(filename))->nodetypes.size();
}

int hil_test_graphs(const std::string &filename) {
  return (parse(filename))->graphs.size();
}

int hil_test_verilogprinter(const std::string &filename) {
  auto nodetypes = parse(filename)->nodetypes;

  std::cout << "------ Verilog RTL-model ------" << std::endl;
  for (const auto *nodetype: nodetypes) {
    auto printer = std::make_unique<VerilogPrinter>(*nodetype);
    std::cout << *printer;
  }

  return 0;
}

TEST(HilTest, SimpleTest) {
  EXPECT_EQ(hil_test("test/hil/test.hil"), 0);
}

TEST(HilTest, NodeTypesTest) {
  EXPECT_EQ(hil_test_nodetypes("test/hil/test.hil"), 6);
}

TEST(HilTest, GraphsTest) {
  EXPECT_EQ(hil_test_graphs("test/hil/test.hil"), 1);
}

TEST(HilTest, VerilogPrinterTest) {
  EXPECT_EQ(hil_test_verilogprinter("test/hil/test.hil"), 0);
}
