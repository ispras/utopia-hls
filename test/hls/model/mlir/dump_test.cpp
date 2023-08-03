//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "HIL/API.h"
#include "HIL/Dumper.h"

#include "HIL/Model.h"
#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"

#include "gtest/gtest.h"

using Model = eda::hls::model::Model;
template<typename Type>
using Transformer = mlir::transforms::Transformer<Type>;

bool dumpToMlirTest(const std::string &filePath) {
  // Construct a model from a '.hil' file.
  const std::string path = getenv("UTOPIA_HLS_HOME") + filePath;
  const Model model = *eda::hls::parser::hil::parse(path).get();
  std::cout << model << std::endl;

  // Dump the model to MLIR.
  Transformer<Model> transformer{model};

  // Check whether one can reconstruct the model. 
  auto modelAfter = transformer.done();
  std::cout << modelAfter << std::endl;
  return true;
}

/* Dumper tests for HIL->MLIR conversion. */
TEST(DumperTest, HilFeedback) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/feedback.hil"), true);
}

TEST(DumperTest, HilIdctRow) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/idct_row.hil"), true);
}

TEST(DumperTest, HilIdct) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/idct.hil"), true);
}

TEST(DumperTest, HilOneKernel) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/one_kernel.hil"), true);
}

TEST(DumperTest, HilSourceSink) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/source_sink.hil"), true);
}

TEST(DumperTest, HilTestInstance) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/test_instance.hil"), true);
}

TEST(DumperTest, HilTestSmall) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/test_small.hil"), true);
}

TEST(DumperTest, HilTest) {
  EXPECT_EQ(dumpToMlirTest("/test/data/hil/test.hil"), true);
}