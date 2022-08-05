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
#include "test_hil_model.h"

#include "gtest/gtest.h"

#include <array>
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>

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

void printModel(const Model &model, const std::string &filename) {
  std::ofstream output(filename);
  eda::hls::model::printDot(output, model);
  output.close();
}

template <typename Balancer>
void prepare(Model &model) {
  Mapper::get().map(model, Library::get());
  std::map<std::string, Parameters> params =
    ParametersOptimizer<Balancer>::get().optimize(criteria, model, indicators);
}

int runLpsolve(Model &model) {
  LatencyLpSolver &solver = LatencyLpSolver::get();
  solver.balance(model, Verbosity::Neutral);
  return solver.getStatus();
}

int lpsolveTest(const std::string &filename) {
  std::shared_ptr<Model> model = parse(filename);
  prepare<LatencyLpSolver>(*model);
  int status = runLpsolve(*model);
  printModel(*model, filename + "_solve.dot");
  return status;
}

int runLpsolve(Model &model, FlowBalanceMode mode) {
  FlowLpSolver &solver = FlowLpSolver::get();
  solver.balance(model, mode, Verbosity::Full);
  return solver.getStatus();
}

int balanceFlowTest(const std::string &filename, FlowBalanceMode mode) {
  std::shared_ptr<Model> model = parse(filename);
  return runLpsolve(*model, mode);
}

template <typename Balancer>
void dijkstraTest(const std::string &filename, LatencyBalanceMode mode) {
  std::shared_ptr<Model> model = parse(filename);
  prepare<Balancer>(*model);

  Balancer::get().balance(*model, mode);

  printModel(*model, filename + "_dijkstra.dot");
}

void topologicalTest(const std::string &filename) {
  std::shared_ptr<Model> model = parse(filename);
  prepare<TopologicalBalancer>(*model);

  TopologicalBalancer::get().balance(*model);

  printModel(*model, filename + "_topological.dot");
}

} // namespace

using QueueBalancer = DijkstraBalancer<std::queue<const Chan*>>;
using PriorityBalancer = DijkstraBalancer<StdPriorityQueue, CompareChan>;

// Hand-written model tests.

TEST(SchedulerTest, ModelTest) {
  Model *model = TestHilModel::get();
  std::cout << *model << std::endl;
}

TEST(SchedulerTest, ModelSolveLatency) {
  Model *model = TestHilModel::get();
  int status = runLpsolve(*model);
  EXPECT_EQ(status, OPTIMAL);
}

TEST(SchedulerTest, ModelASAP) {
  Model *model = TestHilModel::get();
  QueueBalancer::get().balance(*model, LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, ModelALAP) {
  Model *model = TestHilModel::get();
  QueueBalancer::get().balance(*model, LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, ModelPriorityASAP) {
  Model *model = TestHilModel::get();
  PriorityBalancer::get().balance(*model, LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, ModelPriorityALAP) {
  Model *model = TestHilModel::get();
  PriorityBalancer::get().balance(*model, LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, ModelTopological) {
  Model *model = TestHilModel::get();
  TopologicalBalancer::get().balance(*model);
}

// lp_solve flow balancer tests.

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", FlowBalanceMode::Simple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", FlowBalanceMode::Blocking), OPTIMAL);
}

// lp_solve latency balancer tests.

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

// Simple queue-based tests.

TEST(SchedulerTest, SimpleASAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/test.hil", LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, SimpleALAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/test.hil", LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctASAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct.hil", LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctRowASAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct_row.hil", LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctALAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct.hil", LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctRowALAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct_row.hil", LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, SmallASAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/test_small.hil", LatencyBalanceMode::ASAP);
}

// Priority queue-based balancer tests.

TEST(SchedulerTest, SimplePriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/test.hil", LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, SimplePriorityALAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/test.hil", LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctPriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct.hil", LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctRowPriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct_row.hil", LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctPriorityALAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct.hil", LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctRowAPriorityLAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct_row.hil", LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, SmallPriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/test_small.hil", LatencyBalanceMode::ASAP);
}

// Topological balancer tests.

TEST(SchedulerTest, Topological) {
  topologicalTest("test/data/hil/test.hil");
}

TEST(SchedulerTest, IdctTopological) {
  topologicalTest("test/data/hil/idct.hil");
}

TEST(SchedulerTest, IdctRowTopological) {
  topologicalTest("test/data/hil/idct_row.hil");
}

TEST(SchedulerTest, SmallTopological) {
  topologicalTest("test/data/hil/test_small.hil");
}

TEST(SchedulerTest, FeedbackTopological) {
  topologicalTest("test/data/hil/feedback.hil");
}
