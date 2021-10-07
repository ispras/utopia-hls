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
#include "hls/parser/hil/builder.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::model;
using namespace eda::hls::library;
using namespace eda::hls::parser::hil;

std::unique_ptr<Model> hil_parse(const std::string &filename) {
  if (parse(filename) == -1) {
    std::cout << "Could not parse " << filename << std::endl;
    return NULL;
  }

  return Builder::get().create();
}

int hil_test(const std::string &filename) {
  std::cout << *hil_parse(filename);

  return 0;
}

int hil_test_nodetypes(const std::string &filename) {
  return (hil_parse(filename))->nodetypes.size();
}

int hil_test_graphs(const std::string &filename) {
  return (hil_parse(filename))->graphs.size();
}

int hil_test_verilogprinter(const std::string &filename) {
  std::vector<NodeType*> nodetypes = hil_parse(filename)->nodetypes;

  std::cout << "------ Verilog RTL-model ------" << std::endl;
  for (const NodeType *type: nodetypes) {
    VerilogPrinter *printer = new VerilogPrinter(*type);
    std::cout << *printer;
  }

  return 0;
}

TEST(HilTest, SingleTest) {
  EXPECT_EQ(hil_test("test/hil/test.hil"), 0);
}

TEST(HilNodeTypesTest, SingleTest) {
  EXPECT_EQ(hil_test_nodetypes("test/hil/test.hil"), 6);
}

TEST(HilGraphsTest, SingleTest) {
  EXPECT_EQ(hil_test_graphs("test/hil/test.hil"), 1);
}

TEST(HilVerilogPrinterTest, SingleTest) {
  EXPECT_EQ(hil_test_verilogprinter("test/hil/test.hil"), 0);
}
