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
#include "hls/scheduler/dijkstra.h"
#include "hls/scheduler/scheduler.h"
#include "hls/scheduler/solver.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

int lpsolveTest(const std::string &filename, BalanceMode mode) {

  std::unique_ptr<Model> model = parse(filename);
  LpSolver* solver = new LpSolver(model.get());

  solver->balance(mode, Verbosity::Full);

  return solver->getStatus();
}

int dijkstraTest(const std::string &filename) {
  std::unique_ptr<Model> model = parse(filename);
  DijkstraBalancer* balancer = new DijkstraBalancer(model.get());
  balancer->balance();
  return 0;
}

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(lpsolveTest("test/hil/test.hil", BalanceMode::Simple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(lpsolveTest("test/hil/test.hil", BalanceMode::Blocking), OPTIMAL);
}

TEST(SchedulerTest, SolveLatency) {
  EXPECT_EQ(lpsolveTest("test/hil/test.hil", BalanceMode::LatencyLP), OPTIMAL);
}

TEST(SchedulerTest, DijkstraLatency) {
  EXPECT_EQ(dijkstraTest("test/hil/test.hil"), 0);
}

TEST(SchedulerTest, IdctSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/hil/idct.hil", BalanceMode::LatencyLP), OPTIMAL);
}
