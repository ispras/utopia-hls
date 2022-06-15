//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/latency_solver.h"

#include <cassert>
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
  buffers = std::vector<Buffer*>();
  sinks = std::vector<std::string>();
}

void LatencyLpSolver::balance(Model &model, Verbosity verbosity) {
  const Graph *graph = model.main();
  helper.setVerbosity(verbosity);

  balanceLatency(graph);

  // Solve
  helper.printProblem();
  helper.solve();
  helper.printStatus();
  //helper.printResults();

  insertBuffers(model);
  collectGraphTime();
  std::cout << "Max time: " << graphTime << std::endl;

  lastStatus = helper.getStatus();

  // Reset solver for next problem
  helper.reset();
  reset();
}

void LatencyLpSolver::insertBuffers(Model &model) {
  std::vector<double> latencies = helper.getResults();
  unsigned bufsInserted = 0;
  for (const auto *buf : buffers) {
    // lp_solve positions start with 1
    double latency = latencies[buf->position - 1];
    assert(latency >= 0 && ("Delta for channel " + buf->channel->name + " < 0")
      .c_str());
    if (latency != 0) {
      model.insertDelay(*(buf->channel), (unsigned)latency);
      bufsInserted++;
    }
  }
  std::cout << "Total buffers inserted: " << bufsInserted << std::endl;
}

void LatencyLpSolver::collectGraphTime() {
  std::vector<SolverVariable*> sinkVars = helper.getVariables(sinks);
  std::vector<double> latencies = helper.getResults();
  unsigned maxTime = 0;
  for (const auto *sink : sinkVars) {
    unsigned index = sink->columnNumber - 1;
    if (latencies[index] > maxTime) {
      maxTime = (unsigned)latencies[index];
    }
  }
  graphTime = maxTime;
}

void LatencyLpSolver::balanceLatency(const Graph *graph) { 

  for (const Node *node : graph->nodes) {
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
    const std::string &srcName, unsigned srcLatency, Chan *channel) {
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

