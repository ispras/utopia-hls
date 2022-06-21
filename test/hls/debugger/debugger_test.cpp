//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "HIL/Model.h"
#include "hls/parser/hil/parser.h"
#include "hls/debugger/debugger.h"

#include "gtest/gtest.h"

using namespace eda::hls::parser::hil;
using namespace eda::hls::debugger;
using namespace mlir::hil;

bool eqCheckTest(const std::string &fileM, const std::string &fileM2) {

  // Get first model
  mlir::model::MLIRModule modelM = parseToMlir(fileM);
  mlir::hil::Model model = modelM.get_root();

  // Get second model
  mlir::model::MLIRModule modelM2 = parseToMlir(fileM2);
  mlir::hil::Model model2 = modelM2.get_root();

  EqChecker checker = EqChecker::get();

  bool result = checker.equivalent(model, model2);

  return result;
}

/* Equivalence checker tests for model-vs-model_clone pairs. */

// Test for source->sink graph example
TEST(DebuggerTest, SolveSrcSink) {
  EXPECT_EQ(
      eqCheckTest(
          "test/data/hil/source_sink.hil",
          "test/data/hil/source_sink_clone.hil"),
      false);
}

// Test for one kernel example
TEST(DebuggerTest, SolveOneKernel) {
  EXPECT_EQ(
      eqCheckTest(
          "test/data/hil/one_kernel.hil",
          "test/data/hil/one_kernel_clone.hil"),
      false);
}

// Test for no-meaning first HIL example
TEST(DebuggerTest, SolveTest) {
  EXPECT_EQ(
      eqCheckTest("test/data/hil/test.hil",
          "test/data/hil/test_clone.hil"),
      false);
}
