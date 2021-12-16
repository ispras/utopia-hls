//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/dijkstra.h"
#include "hls/scheduler/scheduler.h"
#include "hls/scheduler/solver.h"

#include "gtest/gtest.h"

#include <fstream>
#include <iostream>

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

int lpsolveTest(const std::string &filename, BalanceMode mode) {

  std::shared_ptr<Model> model = parse(filename);
  LpSolver &solver = LpSolver::get();

  solver.balance(*model, mode, Verbosity::Full);
  //std::cout << *model;

  std::ofstream output(filename + "_solve.dot");
  eda::hls::model::printDot(output, *model);
  output.close();


  return solver.getStatus();
}

int dijkstraTest(const std::string &filename, BalanceMode mode) {
  std::shared_ptr<Model> model = parse(filename);
  DijkstraBalancer::get().balance(*model, mode);
  //std::cout << *model;

  std::ofstream output(filename + "_dijkstra.dot");
  eda::hls::model::printDot(output, *model);
  output.close();

  return 0;
}

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil", BalanceMode::FlowSimple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil", BalanceMode::FlowBlocking), OPTIMAL);
}

TEST(SchedulerTest, SolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil", BalanceMode::LatencyLP), OPTIMAL);
}

TEST(SchedulerTest, DijkstraLatencyASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test.hil", BalanceMode::LatencyASAP), 0);
}

TEST(SchedulerTest, DijkstraLatencyALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test.hil", BalanceMode::LatencyALAP), 0);
}

TEST(SchedulerTest, IdctSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/idct.hil", BalanceMode::LatencyLP), OPTIMAL);
}

TEST(SchedulerTest, IdctRowSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/idct_row.hil", BalanceMode::LatencyLP), OPTIMAL);
}

TEST(SchedulerTest, IdctASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct.hil", BalanceMode::LatencyASAP), 0);
}

TEST(SchedulerTest, IdctRowASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct_row.hil", BalanceMode::LatencyASAP), 0);
}

TEST(SchedulerTest, IdctALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct.hil", BalanceMode::LatencyALAP), 0);
}

TEST(SchedulerTest, IdctRowALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct_row.hil", BalanceMode::LatencyALAP), 0);
}

