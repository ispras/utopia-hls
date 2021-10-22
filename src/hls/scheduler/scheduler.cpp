//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===

#include <cassert>
#include <hls/scheduler/scheduler.h>
#include <memory>

namespace eda::hls::scheduler {

std::shared_ptr<double[]> makeCoeffs(const std::vector<std::string> &);
float sumFlows(const std::vector<Argument*> &);

void LpSolver::balance(BalanceMode mode, Verbosity verbosity) {
  helper->setVerbosity(verbosity);

  for (Graph* const graph : model->graphs) {
    
    // Generate a problem to solve
    switch (mode) {
    case Latency:
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
  }
}

void LpSolver::balanceLatency(const Graph* graph) { }

void LpSolver::balanceFlows(BalanceMode mode, const Graph* graph) {
    
    std::vector<std::string> sinks;
    for (Node* const node : graph->nodes) {
      checkFlows(node);
      std::string nodeName = node->name;
      helper->addVariable(nodeName, node);
      genNodeConstraints(nodeName);
      if (node->is_sink()) {
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
    const Binding from = channel->source;
    const Binding to = channel->target;
    std::vector<std::string> names{from.node->name, to.node->name};
    std::vector<double> values{from.port->flow, -1.0 * to.port->flow};

    helper->addConstraint(names, values, type,0);
  }
}

void LpSolver::checkFlows(const Node* node) {
  if (node->is_merge() || node->is_split()) {
    assert(sumFlows(node->type.inputs) == sumFlows(node->type.outputs));
  }
}

float sumFlows(const std::vector<Argument*> &args) {
  float sum = 0;
  for (auto* const arg : args) {
    sum += arg->flow;
  }
  return sum;
}

}  // namespace eda::hls::scheduler

