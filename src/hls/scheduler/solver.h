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
#include "util/singleton.h"

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

struct Buffer final {
  Buffer(Chan *chan, unsigned latency, unsigned position) : channel(chan), 
      latency(latency), position(position) {}

  Chan *channel;
  unsigned latency;
  unsigned position;
};

class LpSolver final : public LatencyBalancer, public Singleton<LpSolver> {

public:

  friend Singleton<LpSolver>;

  ~LpSolver();

  void balance(Model &model, BalanceMode mode, Verbosity verbosity);

  void balance(Model &model) override { 
    balance(model, BalanceMode::LatencyLP, Verbosity::Full); 
  }

  int getStatus() { return lastStatus; }

private:
  LpSolver() : helper(LpSolverHelper::get()), lastStatus(helper.getStatus()) {}
  void deleteBuffers();
  void reset();

  void insertBuffers(Model &model) override;
  void genLatencyConstraints(const std::string &nextName, 
      const std::string &prevName, unsigned latency);
  void genDeltaConstraints(const std::string &dstName, 
      const std::string &srcName, std::vector<std::string> &deltas);
  void genBufferConstraints(const std::string &nextName, 
      const std::string &prevName, unsigned latency, Chan *channel);
  void balanceLatency(const Graph *graph);
  void synchronizeInput(const std::string &varName);

  void checkFlows(const Node *node);
  void balanceFlows(BalanceMode mode, const Graph *graph);
  void genNodeConstraints(const std::string &nodeName);
  void genFlowConstraints(const Graph *graph, OperationType type);
  void collectGraphTime() override;

  const std::string TimePrefix = "t_";
  const std::string FlowPrefix = "f_";
  const std::string DeltaPrefix = "delta_";
  const std::string BufferPrefix = "buf_";
  
  LpSolverHelper &helper;
  int lastStatus;
  std::vector<Buffer*> buffers;
  std::vector<std::string> sinks;
};

} // namespace eda::hls::scheduler
