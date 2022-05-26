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

int callLibraryElementTest(const std::string &name) {
  Parameters params;
  auto metaElement = Library::get().find(name);
  auto element = metaElement->construct(params);
  std::cout << element->ir << std::endl;

  return 0;
}

TEST(LibraryTest, PrintAddTest) {
  EXPECT_EQ(callLibraryElementTest("add1"), 0);
}

