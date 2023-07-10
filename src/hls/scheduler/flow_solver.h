//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the declaration of the FlowLpSolver class  and its
/// supplement structures, that can schedule or balance flows in the 
/// dataflow graph using linear programming.
///
/// \author <a href="mailto:lebedev@ispras.ru">Mikhail Lebedev</a>
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/scheduler/lp_helper.h"
#include "util/singleton.h"

#include <memory>

namespace eda::hls::scheduler {

enum FlowBalanceMode {
  /// Balance the flows, if possible
  Simple,
  
  /// Balance the flows with blockings evaluation
  Blocking,
};

class FlowLpSolver final : public utils::Singleton<FlowLpSolver> {

public:

  friend utils::Singleton<FlowLpSolver>;

  ~FlowLpSolver();

  void balance(model::Model &model, FlowBalanceMode mode, Verbosity verbosity);

  void balance(model::Model &model, FlowBalanceMode mode) { 
    balance(model, mode, Verbosity::Full); 
  }

  int getStatus() const { return lastStatus; }

private:
  FlowLpSolver() : 
      helper(LpSolverHelper::get()), lastStatus(helper.getStatus()) {}
  void reset();

  /// Checks the sum input & output flows of a split/merge node to be equal.
  void checkFlows(const model::Node *node);

  /// Balances the flows of the graph.
  void balanceFlows(FlowBalanceMode mode, const model::Graph *graph);

  /// Generates the constraints for node's flow.
  void genNodeConstraints(const std::string &nodeName);

  /// Generates the constraints for inter-node flows.
  void genFlowConstraints(const model::Graph *graph, OperationType type);
  
  inline std::shared_ptr<double[]> makeCoeffs(
      const std::vector<std::string> &sinks) {
    std::shared_ptr<double[]> sinkCoeffs(new double[sinks.size()]);
      for (unsigned int i = 0; i < sinks.size(); i++) {
        sinkCoeffs[i] = 1.0;
      }
    return sinkCoeffs;
  }

  inline float sumFlows(const std::vector<model::Port*> &ports) {
    float sum = 0;
    for (const auto *port : ports) {
      sum += port->flow;
    }
    return sum;
  }

  LpSolverHelper &helper;
  int lastStatus;
  std::vector<std::string> sinks;
};

} // namespace eda::hls::scheduler