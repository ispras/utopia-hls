//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/scheduler/lp_helper.h"
#include "hls/scheduler/scheduler.h"

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class LpSolver final : public LatencyBalancer {

public:

  LpSolver() : helper(LpSolverHelper::reset()), lastStatus(helper->getStatus()) {}

  ~LpSolver() = default;

  void balance(Model &model, BalanceMode mode, Verbosity verbosity);

  void balance(Model &model) override { 
    balance(model, BalanceMode::LatencyLP, Verbosity::Full); 
  }

  int getStatus() { return lastStatus; }

private:
  void genLatencyConstraints(const std::string &nextName, 
      const std::string &prevName, unsigned latency);
  void genDeltaConstraints(const std::string &dstName, 
      const std::string &srcName, std::vector<std::string> &deltas);
  void genBufferConstraints(const std::string &nextName, 
      const std::string &prevName, unsigned latency, Chan *channel);
  void balanceLatency(const Graph *graph);

  void checkFlows(const Node *node);
  void balanceFlows(BalanceMode mode, const Graph *graph);
  void genNodeConstraints(const std::string &nodeName);
  void genFlowConstraints(const Graph *graph, OperationType type);

  const std::string TimePrefix = "t_";
  const std::string FlowPrefix = "f_";
  const std::string DeltaPrefix = "delta_";
  const std::string BufferPrefix = "buf_";
  
  LpSolverHelper *helper;
  int lastStatus;
};

} // namespace eda::hls::scheduler
