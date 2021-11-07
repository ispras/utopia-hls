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

#include "hls/library/library.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::library;
using namespace eda::hls::parser::hil;

int printVerilogTest(const std::string &filename) {
  auto nodetypes = parse(filename)->nodetypes;

  std::cout << "--- Verilog RTL-model for " << filename << " ---" << std::endl;
  for (const auto *nodetype: nodetypes) {
    auto printer = std::make_unique<VerilogNodeTypePrinter>(*nodetype);
    printer->print(std::cout);
  }

  return 0;
}

int callLibraryElementTest() {
  Parameters params("add");
  auto metaElement = Library::get().find(params.elementName);
  auto element = metaElement->construct(params);
  std::cout << element->ir << std::endl;

  return 0;
}

TEST(HilTest, VerilogNodeTypePrinterTest) {
  EXPECT_EQ(printVerilogTest("test/data/hil/test.hil"), 0);
  EXPECT_EQ(printVerilogTest("test/data/hil/idct.hil"), 0);
  EXPECT_EQ(callLibraryElementTest(), 0);
}

