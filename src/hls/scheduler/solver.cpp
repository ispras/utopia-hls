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
  helper->setVerbosity(verbosity);

  for (Graph* graph : model->graphs) {
    
    // Generate a problem to solve
    switch (mode) {
    case LatencyLinear:
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

    if (mode == LatencyLinear) {
      insertBuffers(graph, helper->getResults());
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
    const Binding from = channel->source;
    const Binding to = channel->target;
    const std::string nextName = to.node->name;
    const std::string prevName = from.node->name;
    std::vector<std::string> names{TimePrefix + nextName, TimePrefix + 
        prevName};
    std::vector<double> values{1.0, -1.0};

    // t_cur >= t_prev + prev_latency
    helper->addConstraint(names, values, OperationType::GreaterOrEqual, 
        from.port->latency);

    const std::string deltaName = DeltaPrefix + nextName + "_" + prevName;
    helper->addVariable(deltaName, nullptr);
    deltas.push_back(deltaName);
    values.push_back(1.0);
    std::vector<std::string> constrNames{deltaName};
    constrNames.push_back(TimePrefix + nextName);
    constrNames.push_back(TimePrefix + prevName);

    // delta_t = t_next - t_prev
    helper->addConstraint(constrNames, values, OperationType::Equal, 0);

    const std::string bufName = BufferPrefix + nextName + "_" + prevName;
    helper->addVariable(bufName, nullptr);
    buffers.push_back(new Buffer/*{from.node, to.node, channel, 
        helper->addVariable(bufName, nullptr)}*/);
    constrNames[0] = bufName;
    helper->addConstraint(constrNames, values, OperationType::Equal, 
        -1.0 * from.port->latency);
  }

  helper->setObjective(deltas, makeCoeffs(deltas).get());
  helper->setMin();

}

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

float sumFlows(const std::vector<Port*> &ports) {
  float sum = 0;
  for (auto* const port : ports) {
    sum += port->flow;
  }
  return sum;
}

}  // namespace eda::hls::scheduler

