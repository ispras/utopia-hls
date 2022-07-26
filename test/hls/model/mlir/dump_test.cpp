//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "HIL/Dumper.h"
#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"

#include "gtest/gtest.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;

bool dumpToMlirTest(const std::string &filePath) {

  const std::string path = getenv("UTOPIA_HOME") + filePath;
  const Model model = *parse(path).get();
  std::stringstream stream;
  dump_model_mlir(model, stream);

  std::cout << stream.str() << std::endl;
  return stream.good();
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