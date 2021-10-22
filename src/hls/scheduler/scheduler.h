//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <hls/model/model.h>
#include <hls/scheduler/lp_helper.h>
#include <string>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

enum BalanceMode {
  Simple,
  Blocking,
  Latency
};

class LpSolver final {

public:

  LpSolver(Model* model_arg) : model(model_arg), helper(new LpSolverHelper) { }

  LpSolver() : helper(new LpSolverHelper) { }

  ~LpSolver() { delete helper; }

  void setModel(Model* model_arg) { model = model_arg; }

  void balance(BalanceMode mode, Verbosity verbosity);

  int getResult() { return helper->getStatus(); }

private:

  void checkFlows(const Node* node);
  void balanceFlows(BalanceMode mode, const Graph* graph);
  void genNodeConstraints(const std::string &nodeName);
  void genFlowConstraints(const Graph* graph, OperationType type);
  void balanceLatency(const Graph* graph);
  
  Model* model;
  LpSolverHelper* helper;

};

} // namespace eda::hls::scheduler
