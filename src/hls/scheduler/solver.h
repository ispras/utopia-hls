//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the declaration of the LpSolver class  and its
/// supplement structures, that can schedule or balance the flows in the 
/// dataflow graph using linear programming.
///
/// \author <a href="mailto:lebedev@ispras.ru">Mikhail Lebedev</a>
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
  void collectGraphTime() override;

  /// Generates the constraint for next node timestep.
  void genLatencyConstraint(const std::string &nextName, 
      const std::string &prevName, unsigned latency);
  
  /// Generates the constraint for time delta between nodes.
  void genDeltaConstraint(const std::string &dstName, 
      const std::string &srcName, std::vector<std::string> &deltas);
  
  /// Generates the constraint for buffer latency.
  void genBufferConstraint(const std::string &nextName, 
      const std::string &prevName, unsigned latency, Chan *channel);

  /// Schedules the graph.
  void balanceLatency(const Graph *graph);

  /// Generates the constraint for input.
  void synchronizeInput(const std::string &varName);

  /// Checks the sum input & output flows of a split/merge node to be equal.
  void checkFlows(const Node *node);

  /// Balances the flows of the graph.
  void balanceFlows(BalanceMode mode, const Graph *graph);

  /// Generates the constraints for node's flow.
  void genNodeConstraints(const std::string &nodeName);

  /// Generates the constraints for inter-node flows.
  void genFlowConstraints(const Graph *graph, OperationType type);
  
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
