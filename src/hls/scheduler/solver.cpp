//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===

#include <cassert>
#include <hls/scheduler/solver.h>
#include <memory>

namespace eda::hls::scheduler {

std::shared_ptr<double[]> makeCoeffs(const std::vector<std::string> &);
float sumFlows(const std::vector<Port*> &);

void LpSolver::balance(BalanceMode mode, Verbosity verbosity) {
  for (Graph* graph : model->graphs) {
    helper->setVerbosity(verbosity);

    // Generate a problem to solve
    switch (mode) {
    case LatencyLP:
      balanceLatency(graph);
      break;
    default:
      balanceFlows(mode, graph);
    }

    // Solve
    helper->printProblem();
    helper->solve();
    helper->printStatus();
    helper->printResults();

    if (mode == LatencyLP) {
      insertBuffers(graph, helper->getResults());
    }
    lastStatus = helper->getStatus();

    // Reset solver for next problem
    if (model->graphs.size() > 1) {
      helper = LpSolverHelper::resetInstance();
    }
  }
}

void LpSolver::balanceLatency(const Graph* graph) { 

  for (Node const* node : graph->nodes) {
    std::string nodeName = node->name;
    helper->addVariable(TimePrefix + nodeName, node);
  }

  std::vector<std::string> deltas;
  for (Chan* const channel : graph->chans) {
    const std::string dstName = channel->target.node->name;
    const std::string srcName = channel->source.node->name;
    unsigned srcLatency = channel->source.port->latency;

    genLatencyConstraints(dstName, srcName, srcLatency);
    genDeltaConstraints(dstName, srcName, deltas);
    genBufferConstraints(dstName, srcName, srcLatency);
  }

  // Minimize deltas
  helper->setObjective(deltas, makeCoeffs(deltas).get());
  helper->setMin();
}

void LpSolver::genLatencyConstraints(const std::string &dstName, 
    const std::string &srcName, unsigned srcLatency) {
  std::vector<std::string> names{TimePrefix + dstName, TimePrefix + srcName};
  std::vector<double> values{1.0, -1.0};

  // t_next >= t_prev + prev_latency
  helper->addConstraint(names, values, OperationType::GreaterOrEqual, srcLatency);
}

void LpSolver::genDeltaConstraints(const std::string &dstName, 
    const std::string &srcName, std::vector<std::string> &deltas) {
  std::vector<double> values{1.0, -1.0, 1.0};
  const std::string deltaName = DeltaPrefix + dstName + "_" + srcName;
  helper->addVariable(deltaName, nullptr);
  deltas.push_back(deltaName);
  std::vector<std::string> constrNames{deltaName, TimePrefix + dstName, 
      TimePrefix + srcName};

  // delta_t = t_next - t_prev
  helper->addConstraint(constrNames, values, OperationType::Equal, 0);
}

void LpSolver::genBufferConstraints(const std::string &dstName, 
    const std::string &srcName, unsigned srcLatency) {
  std::vector<double> values{1.0, -1.0, 1.0};
  const std::string bufName = BufferPrefix + dstName + "_" + srcName;
  std::vector<std::string> constrNames{bufName, TimePrefix + dstName, 
      TimePrefix + srcName};
  helper->addVariable(bufName, nullptr);
  buffers.push_back(new Buffer/*{src.node, dst.node, channel, 
      helper->addVariable(bufName, nullptr)}*/);
  // buf_next_prev = t_next - (t_prev + prev_latency)
  helper->addConstraint(constrNames, values, OperationType::Equal, 
      -1.0 * srcLatency);
}

void LpSolver::balanceFlows(BalanceMode mode, const Graph* graph) {
    
    std::vector<std::string> sinks;
    for (Node* const node : graph->nodes) {
      checkFlows(node);
      std::string nodeName = node->name;
      helper->addVariable(nodeName, node);
      genNodeConstraints(nodeName);
      if (node->isSink()) {
        sinks.push_back(node->name);
      }     
    }

    // Add constraints for channels
    if (mode == Simple) {
      // flow_src*coeff_src == flow_dst*coeff_dst
      genFlowConstraints(graph, OperationType::Equal);
    }

    if (mode == Blocking) {
      // flow_src*coeff_src >= flow_dst*coeff_dst
      genFlowConstraints(graph, OperationType::GreaterOrEqual);
    }

    // Maximize sink flow
    helper->setObjective(sinks, makeCoeffs(sinks).get());
    helper->setMax();
}

std::shared_ptr<double[]> makeCoeffs(const std::vector<std::string> &sinks) {
  std::shared_ptr<double[]> sinkCoeffs(new double[sinks.size()]);
    for (unsigned int i = 0; i < sinks.size(); i++) {
      sinkCoeffs[i] = 1.0;
    }
  return sinkCoeffs;
}

void LpSolver::genNodeConstraints(const std::string &nodeName) {
  std::vector<std::string> names{nodeName};
  std::vector<double> valOne{1.0};

  // Add 'coeff >= 0.01'
  helper->addConstraint(names, valOne, OperationType::GreaterOrEqual, 0.01);
  // Add 'coeff <= 1'
  helper->addConstraint(names, valOne, OperationType::LessOrEqual, 1);
}

void LpSolver::genFlowConstraints(const Graph* graph, OperationType type) {
  for (Chan* const channel : graph->chans) {
    const Binding src = channel->source;
    const Binding dst = channel->target;
    std::vector<std::string> names{src.node->name, dst.node->name};
    std::vector<double> values{src.port->flow, -1.0 * dst.port->flow};

    helper->addConstraint(names, values, type,0);
  }
}

void LpSolver::checkFlows(const Node* node) {
  if (node->isMerge() || node->isSplit()) {
    assert(sumFlows(node->type.inputs) == sumFlows(node->type.outputs));
  }
}

float sumFlows(const std::vector<Port*> &ports) {
  float sum = 0;
  for (auto* const port : ports) {
    sum += port->flow;
  }
  return sum;
}

}  // namespace eda::hls::scheduler

