//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/dijkstra.h"
#include "hls/scheduler/latency_solver.h"
#include "hls/scheduler/flow_solver.h"

#include "gtest/gtest.h"

#include <fstream>
#include <iostream>

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

int lpsolveTest(const std::string &filename) {

  std::shared_ptr<Model> model = parse(filename);
  LatencyLpSolver &solver = LatencyLpSolver::get();

  solver.balance(*model, Verbosity::Full);
  //std::cout << *model;

  std::ofstream output(filename + "_solve.dot");
  eda::hls::model::printDot(output, *model);
  output.close();

  return solver.getStatus();
}

int balanceFlowTest(const std::string &filename, FlowBalanceMode mode) {
  std::shared_ptr<Model> model = parse(filename);
  FlowLpSolver &solver = FlowLpSolver::get();
  solver.balance(*model, mode, Verbosity::Full);
  return solver.getStatus();
}

int dijkstraTest(const std::string &filename, LatencyBalanceMode mode) {
  std::shared_ptr<Model> model = parse(filename);
  DijkstraBalancer::get().balance(*model, mode);
  //std::cout << *model;

  std::ofstream output(filename + "_dijkstra.dot");
  eda::hls::model::printDot(output, *model);
  output.close();

  return 0;
}

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", FlowBalanceMode::Simple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", FlowBalanceMode::Blocking), OPTIMAL);
}

TEST(SchedulerTest, SolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil"), OPTIMAL);
}

TEST(SchedulerTest, DijkstraLatencyASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test.hil", LatencyBalanceMode::ASAP), 0);
}

TEST(SchedulerTest, DijkstraLatencyALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test.hil", LatencyBalanceMode::ALAP), 0);
}

TEST(SchedulerTest, IdctSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/idct.hil"), OPTIMAL);
}

TEST(SchedulerTest, IdctRowSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/idct_row.hil"), OPTIMAL);
}

TEST(SchedulerTest, IdctASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct.hil", LatencyBalanceMode::ASAP), 0);
}

TEST(SchedulerTest, IdctRowASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct_row.hil", LatencyBalanceMode::ASAP), 0);
}

TEST(SchedulerTest, IdctALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct.hil", LatencyBalanceMode::ALAP), 0);
}

TEST(SchedulerTest, IdctRowALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/idct_row.hil", LatencyBalanceMode::ALAP), 0);
}

