//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/scheduler.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

int lpsolveTest(const std::string &filename, BalanceMode mode) {

  std::unique_ptr<Model> model = parse(filename);
  LpSolver* solver = new LpSolver();
  solver->setModel(model.get());

  solver->balance(mode, Verbosity::Full);

  return solver->getResult();
}

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(lpsolveTest("test/hil/test.hil", BalanceMode::Simple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(lpsolveTest("test/hil/test.hil", BalanceMode::Blocking), OPTIMAL);
}
