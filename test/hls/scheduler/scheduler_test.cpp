//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"
#include "hls/mapper/mapper.h"
#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/dijkstra.h"
#include "hls/scheduler/flow_solver.h"
#include "hls/scheduler/latency_balancer_base.h"
#include "hls/scheduler/latency_solver.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/topological_balancer.h"

#include "gtest/gtest.h"

#include <fstream>
#include <iostream>

using namespace eda::hls::library;
using namespace eda::hls::mapper;
using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

namespace {

Indicators indicators;
// Optimization criterion and constraints.
eda::hls::model::Criteria criteria(
  PERF,
  eda::hls::model::Constraint<unsigned>(40000, 500000),                                // Frequency (kHz)
  eda::hls::model::Constraint<unsigned>(1000,  500000),                                // Performance (=frequency)
  eda::hls::model::Constraint<unsigned>(0,     1000),                                  // Latency (cycles)
  eda::hls::model::Constraint<unsigned>(),  // Power (does not matter)
  eda::hls::model::Constraint<unsigned>(1,     10000000));                             // Area (number of LUTs)

int lpsolveTest(const std::string &filename) {
  std::shared_ptr<Model> model = parse(filename);
  Mapper::get().map(*model, Library::get());
  ParametersOptimizer::get().setBalancer(&LatencyLpSolver::get());

  std::map<std::string, Parameters> params =
    ParametersOptimizer::get().optimize(criteria, *model, indicators);

  LatencyLpSolver &solver = LatencyLpSolver::get();
  solver.balance(*model, Verbosity::Neutral);

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
  Mapper::get().map(*model, Library::get());
  ParametersOptimizer::get().setBalancer(&DijkstraBalancer::get());
  std::map<std::string, Parameters> params =
    ParametersOptimizer::get().optimize(criteria, *model, indicators);

  DijkstraBalancer::get().balance(*model, mode);

  std::ofstream output(filename + "_dijkstra.dot");
  eda::hls::model::printDot(output, *model);
  output.close();

  return 0;
}

int topologicalTest(const std::string &filename) {
  std::shared_ptr<Model> model = parse(filename);
  Mapper::get().map(*model, Library::get());
  ParametersOptimizer::get().setBalancer(&TopologicalBalancer::get());
  std::map<std::string, Parameters> params =
    ParametersOptimizer::get().optimize(criteria, *model, indicators);

  TopologicalBalancer::get().balance(*model);

  std::ofstream output(filename + "_topological.dot");
  eda::hls::model::printDot(output, *model);
  output.close();

  return 0;
}

} // namespace

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", FlowBalanceMode::Simple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", FlowBalanceMode::Blocking), OPTIMAL);
}

TEST(SchedulerTest, SolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test.hil"), OPTIMAL);
}

TEST(SchedulerTest, IdctSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/idct.hil"), OPTIMAL);
}

TEST(SchedulerTest, IdctRowSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/idct_row.hil"), OPTIMAL);
}

TEST(SchedulerTest, SmallSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/test_small.hil"), OPTIMAL);
}

TEST(SchedulerTest, FeedbackSolveLatency) {
  EXPECT_EQ(lpsolveTest("test/data/hil/feedback.hil"), INFEASIBLE);
}

TEST(SchedulerTest, DijkstraLatencyASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test.hil", LatencyBalanceMode::ASAP), 0);
}

TEST(SchedulerTest, DijkstraLatencyALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test.hil", LatencyBalanceMode::ALAP), 0);
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

TEST(SchedulerTest, SmallASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/test_small.hil", LatencyBalanceMode::ASAP), 0);
}

/*TEST(SchedulerTest, FeedbackASAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/feedback.hil", LatencyBalanceMode::ASAP), 0);
}*/

/*TEST(SchedulerTest, FeedbackALAP) {
  EXPECT_EQ(dijkstraTest("test/data/hil/feedback.hil", LatencyBalanceMode::ALAP), 0);
}*/

TEST(SchedulerTest, Topological) {
  EXPECT_EQ(topologicalTest("test/data/hil/test.hil"), 0);
}

TEST(SchedulerTest, IdctTopological) {
  EXPECT_EQ(topologicalTest("test/data/hil/idct.hil"), 0);
}

TEST(SchedulerTest, IdctRowTopological) {
  EXPECT_EQ(topologicalTest("test/data/hil/idct_row.hil"), 0);
}

TEST(SchedulerTest, SmallTopological) {
  EXPECT_EQ(topologicalTest("test/data/hil/test_small.hil"), 0);
}

TEST(SchedulerTest, FeedbackTopological) {
  EXPECT_EQ(topologicalTest("test/data/hil/feedback.hil"), 0);
}
