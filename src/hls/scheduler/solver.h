//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <hls/scheduler/lp_helper.h>
#include <hls/scheduler/scheduler.h>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class LpSolver final : public LatencyBalancer {

public:

  LpSolver(Model* modelArg) : LatencyBalancer(modelArg), helper(LpSolverHelper::resetInstance()) { }

  LpSolver() : helper(LpSolverHelper::resetInstance()) { }

  ~LpSolver() { }

  void balance(BalanceMode mode, Verbosity verbosity);

  void balance() override { balance(BalanceMode::LatencyLinear, Verbosity::Full); }

  int getResult() { return helper->getStatus(); }

private:

  void checkFlows(const Node* node);
  void balanceFlows(BalanceMode mode, const Graph* graph);
  void genNodeConstraints(const std::string &nodeName);
  void genFlowConstraints(const Graph* graph, OperationType type);
  void balanceLatency(const Graph* graph);

  //NodeType* findType(const std::string &name);

  const std::string TimePrefix = "t_";
  const std::string FlowPrefix = "f_";
  const std::string DeltaPrefix = "delta_";
  const std::string BufferPrefix = "buf_";
  
  LpSolverHelper* helper;
};

} // namespace eda::hls::scheduler
