//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/flow_solver.h"

#include <cassert>
#include <iostream>

namespace eda::hls::scheduler {

FlowLpSolver::~FlowLpSolver() { 
  helper.reset(); 
}

void FlowLpSolver::reset() {
  sinks = std::vector<std::string>();
}

void FlowLpSolver::balance(Model &model, FlowBalanceMode mode, 
    Verbosity verbosity) {
  const Graph *graph = model.main();
  helper.setVerbosity(verbosity);

  balanceFlows(mode, graph);

  // Solve
  helper.printProblem();
  helper.solve();
  helper.printStatus();
  //helper.printResults();

  lastStatus = helper.getStatus();

  // Reset solver for next problem
  helper.reset();
  reset();
}

void FlowLpSolver::balanceFlows(FlowBalanceMode mode, const Graph *graph) {
  
  for (const auto *node : graph->nodes) {
    checkFlows(node);
    std::string nodeName = node->name;
    helper.addVariable(nodeName, node);
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
  helper.setObjective(sinks, makeCoeffs(sinks).get());
  helper.setMax();
}

void FlowLpSolver::genNodeConstraints(const std::string &nodeName) {
  std::vector<std::string> names{nodeName};
  std::vector<double> valOne{1.0};

  // Add 'coeff >= 0.01'
  helper.addConstraint(names, valOne, OperationType::GreaterOrEqual, 0.01);
  // Add 'coeff <= 1'
  helper.addConstraint(names, valOne, OperationType::LessOrEqual, 1);
}

void FlowLpSolver::genFlowConstraints(const Graph *graph, OperationType type) {
  for (const auto* channel : graph->chans) {
    const Binding src = channel->source;
    const Binding dst = channel->target;
    std::vector<std::string> names{src.node->name, dst.node->name};
    std::vector<double> values{src.port->flow, -1.0 * dst.port->flow};

    helper.addConstraint(names, values, type,0);
  }
}

void FlowLpSolver::checkFlows(const Node *node) {
  if (node->isMerge() || node->isSplit()) {
    assert((sumFlows(node->type.inputs) == sumFlows(node->type.outputs))
      && ("Input & output flows for " + node->name + " do not match!").c_str());
  }
}

}  // namespace eda::hls::scheduler

