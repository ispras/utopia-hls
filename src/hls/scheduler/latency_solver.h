//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the declaration of the LatencyLpSolver class  and its
/// supplement structures, that can schedule or balance latencies in the
/// dataflow graph using linear programming.
///
/// \author <a href="mailto:lebedev@ispras.ru">Mikhail Lebedev</a>
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/scheduler/latency_balancer_base.h"
#include "hls/scheduler/lp_helper.h"
#include "util/singleton.h"

#include <memory>

using namespace eda::hls::model;
using namespace eda::util;

namespace eda::hls::scheduler {

struct Buffer final {
  Buffer(model::Chan *chan, int latency, int position) : channel(chan),
      latency(latency), position(position) {}

  model::Chan *channel;
  int latency;
  int position;
};

class LatencyLpSolver final : public LatencyBalancerBase, public Singleton<LatencyLpSolver> {

public:

  friend Singleton<LatencyLpSolver>;

  ~LatencyLpSolver();

  void balance(model::Model &model, Verbosity verbosity);

  void balance(Model &model) override { 
    balance(model, Verbosity::Neutral); 
  }

  int getStatus() const { return lastStatus; }

private:
  LatencyLpSolver() : helper(LpSolverHelper::get()), lastStatus(helper.getStatus()) {}
  void deleteBuffers();
  void reset();

  void insertBuffers(model::Model &model) override;
  void collectGraphTime(const model::Graph &graph);

  /// Generates the constraint for next node timestep.
  void genLatencyConstraint(const std::string &nextName,
      const std::string &prevName, unsigned latency);

  /// Generates the constraint for time delta between nodes.
  void genDeltaConstraint(const std::string &dstName,
      const std::string &srcName, std::vector<std::string> &deltas);

  /// Generates the constraint for buffer latency.
  void genBufferConstraint(const std::string &nextName,
      const std::string &prevName, unsigned latency, model::Chan *channel);

  /// Schedules the graph.
  void balanceLatency(const model::Graph *graph);

  /// Generates the constraint for input.
  void synchronizeInput(const std::string &varName);

  inline std::shared_ptr<double[]> makeCoeffs(const std::vector<std::string> &sinks) {
    std::shared_ptr<double[]> sinkCoeffs(new double[sinks.size()]);
      for (unsigned int i = 0; i < sinks.size(); i++) {
        sinkCoeffs[i] = 1.0;
      }
    return sinkCoeffs;
  }

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
