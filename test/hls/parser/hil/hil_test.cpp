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
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;

int hilTest(const std::string &filename) {
  std::shared_ptr<Model> model = parse(filename);

  if (!model)
    return -1;

  std::cout << *model;
  printDot(std::cout, *model);

  return 0;
}

int hilTestNodeTypes(const std::string &filename) {
  return (parse(filename))->nodetypes.size();
}

int hilTestGraphs(const std::string &filename) {
  return (parse(filename))->graphs.size();
}

TEST(HilTest, SimpleTest) {
  EXPECT_EQ(hilTest("test/data/hil/test.hil"), 0);
  EXPECT_EQ(hilTest("test/data/hil/idct.hil"), 0);
}

TEST(HilTest, NodeTypesTest) {
  EXPECT_EQ(hilTestNodeTypes("test/data/hil/test.hil"), 6);
}

TEST(HilTest, GraphsTest) {
  EXPECT_EQ(hilTestGraphs("test/data/hil/test.hil"), 1);
}

