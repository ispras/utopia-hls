//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/dijkstra.h"
#include "hls/scheduler/scheduler.h"
#include "hls/scheduler/solver.h"

#include "gtest/gtest.h"

#include <iostream>

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

int lpsolveTest(const std::string &filename, BalanceMode mode) {

  std::unique_ptr<Model> model = parse(filename);
  LpSolver &solver = LpSolver::get();

  solver.balance(*model, mode, Verbosity::Full);
  //std::cout << *model;
  return solver.getStatus();
}

int dijkstraTest(const std::string &filename) {
  std::unique_ptr<Model> model = parse(filename);
  DijkstraBalancer::get().balance(*model);
  //std::cout << *model;
  return 0;
}

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil", BalanceMode::Simple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil", BalanceMode::Blocking), OPTIMAL);
}

TEST(SchedulerTest, SolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil", BalanceMode::LatencyLP), OPTIMAL);
}

TEST(SchedulerTest, DijkstraLatency) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test.hil"), 0);
}

TEST(SchedulerTest, IdctSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/idct.hil", BalanceMode::LatencyLP), OPTIMAL);
}

TEST(SchedulerTest, IdctDijkstraLatency) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct.hil"), 0);
}
