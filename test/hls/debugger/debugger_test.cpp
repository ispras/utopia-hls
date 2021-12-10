//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"
#include "hls/debugger/debugger.h"

#include "gtest/gtest.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::debugger;

bool eqCheckTest(const std::string &fileM, const std::string &fileM2) {

  std::unique_ptr<Model> modelM = parse(fileM);
  std::unique_ptr<Model> modelM2 = parse(fileM2);
  Verifier verifier = Verifier::get();

  return verifier.equivalent(*modelM.get(), *modelM2.get());
}

TEST(DebuggerTest, SolveKernel) {
  EXPECT_EQ(
      eqCheckTest(
          "test/data/hil/one_kernel.hil",
          "test/data/hil/one_kernel_clone.hil"),
      false);
}

TEST(DebuggerTest, SolveTest) {
  EXPECT_EQ(
      eqCheckTest("test/data/hil/test.hil",
          "test/data/hil/test_clone.hil"),
      false);
}

