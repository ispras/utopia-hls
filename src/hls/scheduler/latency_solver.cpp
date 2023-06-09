//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/latency_solver.h"
#include "utils/assert.h"

#include <iostream>

namespace eda::hls::scheduler {

LatencyLpSolver::~LatencyLpSolver() { 
  deleteBuffers();
  helper.reset(); 
}

void LatencyLpSolver::deleteBuffers() {
  for (const auto *buf : buffers) {
    delete buf;
  }
}

void LatencyLpSolver::reset() {
  deleteBuffers();
  buffers.clear();
  sinks.clear();
}

void LatencyLpSolver::balance(model::Model &model, Verbosity verbosity) {
  LatencyBalancerBase::init();
  const model::Graph *graph = model.main();
  helper.setVerbosity(verbosity);

  balanceLatency(graph);

  // Solve
  if (verbosity > 3) {
    helper.printProblem();
  }
  helper.solve();
  helper.printStatus();
  if (verbosity > 4) {
    helper.printResults();
  }
  
  lastStatus = helper.getStatus();
  if (lastStatus == 0 || lastStatus == 1) {
    insertBuffers(model);
    collectGraphTime(*graph);
    printGraphTime();
  }
  
  // Reset solver for next problem
  helper.reset();
  reset();
}

void LatencyLpSolver::insertBuffers(model::Model &model) {
  std::vector<double> latencies = helper.getResults();
  unsigned bufsInserted = 0;
  unsigned totalDelta = 0;
  for (const auto *buf : buffers) {
    // lp_solve positions start with 1
    int position = buf->position - 1;
    uassert(position >= 0, "Position can't be negative!\n");
    double latency = latencies[position];
    uassert(latency >= 0, "Delta for channel " + buf->channel->name + " < 0");
    if (latency != 0) {
      buf->channel->latency = (unsigned)latency;
      model.insertDelay(*(buf->channel), (unsigned)latency);
      bufsInserted++;
      totalDelta+=(unsigned)latency;
    }
  }
  std::cout << "Total buffers inserted: " << bufsInserted << std::endl;
  std::cout << "Total buffers capacity: " << totalDelta << std::endl;
}

void LatencyLpSolver::collectGraphTime(const model::Graph &graph) {
  std::vector<SolverVariable*> sinkVars = helper.getVariables(sinks);
  std::vector<double> latencies = helper.getResults();
  for (const auto *sink : sinkVars) {
    unsigned index = sink->columnNumber - 1;
    graphTime = std::max((unsigned)latencies[index], graphTime);
  }
}

void LatencyLpSolver::balanceLatency(const model::Graph *graph) { 

  for (const auto *node : graph->nodes) {
    std::string varName = TimePrefix + node->name;
    helper.addVariable(varName, node);
    
    if (node->isSink()) {
      sinks.push_back(varName);
    }

    /// Agreement: inputs have to be synchronized (start_time = 0)
    if (node->isSource()) {
      synchronizeInput(varName);
    }
  }

  std::vector<std::string> deltas;
  for (auto *channel : graph->chans) {
    const std::string dstName = channel->target.node->name;
    const std::string srcName = channel->source.node->name;
    unsigned srcLatency = channel->ind.ticks;

    genLatencyConstraint(dstName, srcName, srcLatency);
    genDeltaConstraint(dstName, srcName, deltas);
    genBufferConstraint(dstName, srcName, srcLatency, channel);
  }

  // Minimize deltas
  helper.setObjective(deltas, makeCoeffs(deltas).get());
  helper.setMin();
}

void LatencyLpSolver::synchronizeInput(const std::string &varName) {
  std::vector<std::string> name{varName};
  std::vector<double> value{1.0};

  // t_source = 0
  helper.addConstraint(name, value, OperationType::Equal, 0);
}

void LatencyLpSolver::genLatencyConstraint(const std::string &dstName, 
    const std::string &srcName, unsigned srcLatency) {
  std::vector<std::string> names{TimePrefix + dstName, TimePrefix + srcName};
  std::vector<double> values{1.0, -1.0};

  // t_next >= t_prev + prev_latency
  helper.addConstraint(names, values, OperationType::GreaterOrEqual, 
      srcLatency);
}

void LatencyLpSolver::genDeltaConstraint(const std::string &dstName, 
    const std::string &srcName, std::vector<std::string> &deltas) {
  std::vector<double> values{1.0, -1.0, 1.0};
  const std::string deltaName = DeltaPrefix + dstName + "_" + srcName;
  helper.addVariable(deltaName, nullptr);
  deltas.push_back(deltaName);
  std::vector<std::string> constrNames{deltaName, TimePrefix + dstName, 
      TimePrefix + srcName};

  // delta_t = t_next - t_prev
  helper.addConstraint(constrNames, values, OperationType::Equal, 0);
}

void LatencyLpSolver::genBufferConstraint(const std::string &dstName, 
    const std::string &srcName, unsigned srcLatency, model::Chan *channel) {
  std::vector<double> values{1.0, -1.0, 1.0};
  const std::string bufName = BufferPrefix + dstName + "_" + srcName;
  std::vector<std::string> constrNames{bufName, TimePrefix + dstName, 
      TimePrefix + srcName};
  SolverVariable *bufferVariable = helper.addVariable(bufName, nullptr);
  buffers.push_back(new Buffer{channel, 0, bufferVariable->columnNumber});
  
  // buf_next_prev = t_next - (t_prev + prev_latency)
  helper.addConstraint(constrNames, values, OperationType::Equal, 
      -1.0 * srcLatency);
}

}  // namespace eda::hls::scheduler

