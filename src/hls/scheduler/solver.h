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

struct Buffer final {
  Buffer(Chan *chan, unsigned latency, unsigned position) : channel(chan), 
      latency(latency), position(position) {}

  Chan *channel;
  unsigned latency;
  unsigned position;
};

class LpSolver final : public LatencyBalancer {

public:

  LpSolver() : helper(LpSolverHelper::get()), lastStatus(helper.getStatus()) {}

  ~LpSolver() { helper.reset(); }

  void balance(Model &model, BalanceMode mode, Verbosity verbosity);

  void balance(Model &model) override { 
    balance(model, BalanceMode::LatencyLP, Verbosity::Full); 
  }

  int getStatus() { return lastStatus; }

private:
  void insertBuffers(Model &model) override;
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
  
  LpSolverHelper &helper;
  int lastStatus;
  std::vector<Buffer*> buffers;
};

} // namespace eda::hls::scheduler
