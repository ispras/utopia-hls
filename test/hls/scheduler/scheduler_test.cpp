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

namespace tool = eda::hls;
namespace mdl = tool::model;
namespace sched = tool::scheduler;

namespace {

mdl::Indicators indicators;
// Optimization criterion and constraints.
mdl::Criteria criteria(
  Indicator::PERF,
  mdl::Constraint<unsigned>(40000, 500000),       // Frequency (kHz)
  mdl::Constraint<unsigned>(1000,  500000),       // Performance (=frequency)
  mdl::Constraint<unsigned>(0,     1000),         // Latency (cycles)
  mdl::Constraint<unsigned>(),                    // Power (does not matter)
  mdl::Constraint<unsigned>(1,     10000000));    // Area (number of LUTs)

void printModel(const mdl::Model &model, const std::string &filename) {
  std::ofstream output(filename);
  mdl::printDot(output, model);
  output.close();
}

template <typename Balancer>
void prepare(mdl::Model &model) {
  tool::mapper::Mapper::get().map(model, tool::library::Library::get());
  std::map<std::string, mdl::Parameters> params = 
    sched::ParametersOptimizer<Balancer>::get()
      .optimize(criteria, model, indicators);
}

int runLpsolve(mdl::Model &model) {
  sched::LatencyLpSolver &solver = sched::LatencyLpSolver::get();
  solver.balance(model, sched::Verbosity::Neutral);
  return solver.getStatus();
}

int lpsolveTest(const std::string &filename) {
  std::shared_ptr<mdl::Model> model = tool::parser::hil::parse(filename);
  prepare<sched::LatencyLpSolver>(*model);
  int status = runLpsolve(*model);
  printModel(*model, filename + "_solve.dot");
  return status;
}

int runLpsolve(mdl::Model &model, sched::FlowBalanceMode mode) {
  sched::FlowLpSolver &solver = sched::FlowLpSolver::get();
  solver.balance(model, mode, sched::Verbosity::Full);
  return solver.getStatus();
}

int balanceFlowTest(const std::string &filename, sched::FlowBalanceMode mode) {
  std::shared_ptr<mdl::Model> model = tool::parser::hil::parse(filename);
  return runLpsolve(*model, mode);
}

template <typename Balancer>
void dijkstraTest(const std::string &filename, sched::LatencyBalanceMode mode) {
  std::shared_ptr<mdl::Model> model = tool::parser::hil::parse(filename);
  prepare<Balancer>(*model);

  Balancer::get().balance(*model, mode);

  printModel(*model, filename + "_dijkstra.dot");
}

void topologicalTest(const std::string &filename) {
  std::shared_ptr<mdl::Model> model = tool::parser::hil::parse(filename);
  prepare<sched::TopologicalBalancer>(*model);

  sched::TopologicalBalancer::get().balance(*model);

  printModel(*model, filename + "_topological.dot");
}

} // namespace

using QueueBalancer = sched::DijkstraBalancer<std::queue<const mdl::Chan*>>;
using PriorityBalancer 
    = sched::DijkstraBalancer<sched::StdPriorityQueue, sched::CompareChan>;

// Hand-written model tests.

TEST(SchedulerTest, ModelTest) {
  mdl::Model *model = TestHilModel::get();
  std::cout << *model << std::endl;
}

TEST(SchedulerTest, ModelSolveLatency) {
  mdl::Model *model = TestHilModel::get();
  int status = runLpsolve(*model);
  EXPECT_EQ(status, OPTIMAL);
}

TEST(SchedulerTest, ModelASAP) {
  mdl::Model *model = TestHilModel::get();
  QueueBalancer::get().balance(*model, sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, ModelALAP) {
  mdl::Model *model = TestHilModel::get();
  QueueBalancer::get().balance(*model, sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, ModelPriorityASAP) {
  mdl::Model *model = TestHilModel::get();
  PriorityBalancer::get().balance(*model, sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, ModelPriorityALAP) {
  mdl::Model *model = TestHilModel::get();
  PriorityBalancer::get().balance(*model, sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, ModelTopological) {
  mdl::Model *model = TestHilModel::get();
  sched::TopologicalBalancer::get().balance(*model);
}

// lp_solve flow balancer tests.

TEST(SchedulerTest, SolveSimpleInfeasible) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", 
    sched::FlowBalanceMode::Simple), INFEASIBLE);
}

TEST(SchedulerTest, SolveBlocking) {
  EXPECT_EQ(balanceFlowTest("test/data/hil/test.hil", 
    sched::FlowBalanceMode::Blocking), OPTIMAL);
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
  dijkstraTest<QueueBalancer>("test/data/hil/test.hil", 
    sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, SimpleALAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/test.hil", 
    sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctASAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct.hil", 
    sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctRowASAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct_row.hil", 
    sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctALAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct.hil", 
    sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctRowALAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/idct_row.hil", 
    sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, SmallASAP) {
  dijkstraTest<QueueBalancer>("test/data/hil/test_small.hil", 
    sched::LatencyBalanceMode::ASAP);
}

// Priority queue-based balancer tests.

TEST(SchedulerTest, SimplePriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/test.hil", 
    sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, SimplePriorityALAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/test.hil", 
    sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctPriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct.hil", 
    sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctRowPriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct_row.hil", 
    sched::LatencyBalanceMode::ASAP);
}

TEST(SchedulerTest, IdctPriorityALAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct.hil", 
    sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, IdctRowAPriorityLAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/idct_row.hil", 
    sched::LatencyBalanceMode::ALAP);
}

TEST(SchedulerTest, SmallPriorityASAP) {
  dijkstraTest<PriorityBalancer>("test/data/hil/test_small.hil", 
    sched::LatencyBalanceMode::ASAP);
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
