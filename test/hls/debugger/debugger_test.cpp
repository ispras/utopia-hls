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

  std::shared_ptr<mlir::hil::Model> modelM = parseToMlir(fileM);
  std::shared_ptr<mlir::hil::Model> modelM2 = parseToMlir(fileM2);
  EqChecker checker = EqChecker::get();

  return checker.equivalent(*modelM.get(), *modelM2.get());
}

/*TEST(DebuggerTest, SolveKernel) {
  EXPECT_EQ(
      eqCheckTest(
          "test/data/hil/one_kernel.hil",
          "test/data/hil/one_kernel_clone.hil"),
      false);
}*/

TEST(DebuggerTest, SolveTest) {
  EXPECT_EQ(
      eqCheckTest("test/data/hil/test.hil",
          "test/data/hil/test_clone.hil"),
      false);
}

