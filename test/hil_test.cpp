//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "gtest/gtest.h"

#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;

int hil_test(const std::string &filename) {
  if (parse(filename) == -1) {
    std::cout << "Could not parse " << filename << std::endl;
    return -1;
  }

  std::unique_ptr<Model> model = Builder::get().create();
  std::cout << *model;

  return 0;
}

TEST(HilTest, SingleTest) {
  EXPECT_EQ(hil_test("test/hil/test.hil"), 0);
}
