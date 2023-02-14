//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "HIL/Model.h"
#include "hls/debugger/eq_checker.h"
#include "hls/parser/hil/parser.h"

#include "gtest/gtest.h"

bool eqCheckTest(const std::string &fileM, const std::string &fileM2) {

  using EqChecker = eda::hls::eqchecker::EqChecker;
  using MLIRModule = mlir::model::MLIRModule;
  using Model = mlir::hil::Model;

  // Get first model for equivalence checking.
  const std::string pathM = getenv("UTOPIA_HOME") + fileM;
  MLIRModule modelM = eda::hls::parser::hil::parseToMlir(pathM);
  Model model = modelM.get_root();

  // Get second model for equivalence checking.
  const std::string pathM2 = getenv("UTOPIA_HOME") + fileM2;
  MLIRModule modelM2 = eda::hls::parser::hil::parseToMlir(pathM2);
  Model model2 = modelM2.get_root();

  // Perform checking itself.
  EqChecker checker = EqChecker::get();

  bool result = checker.equivalent(model, model2);

  return result;
}

/* Equivalence checker tests for model-vs-model_clone pairs. */

// Test for source->sink graph example.
TEST(EqCheckerTestSuite, solveSrcSink) {
  EXPECT_EQ(
      eqCheckTest(
          "/test/data/hil/source_sink.hil",
          "/test/data/hil/source_sink_clone.hil"),
      false);
}

// Test for one-kernel example.
TEST(EqCheckerTestSuite, solveOneKernel) {
  EXPECT_EQ(
      eqCheckTest(
          "/test/data/hil/one_kernel.hil",
          "/test/data/hil/one_kernel_clone.hil"),
      false);
}

// Test for split-merge HIL example.
TEST(EqCheckerTestSuite, solveTest) {
  EXPECT_EQ(
      eqCheckTest(
          "/test/data/hil/test.hil",
          "/test/data/hil/test_clone.hil"),
      false);
}
