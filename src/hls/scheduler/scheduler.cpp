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

double *makeCoeffs(const std::vector<std::string> &);
float sumFlows(const std::vector<Argument*> &);

void LpSolver::balance() {

  for (const Graph* graph : model->graphs) {
    std::vector<std::string> sinks;
    
    for (Node* node : graph->nodes) {
      checkFlows(node);
      std::string nodeName = node->name;
      helper->addVariable(nodeName, node);

      std::vector<std::string> names{nodeName};
      std::vector<double> valOne{1.0};

      // Add 'coeff >= 0.01'
      helper->addConstraint(names, valOne, OperationType::GreaterOrEqual, 0.01);
      // Add 'coeff <= 1'
      helper->addConstraint(names, valOne, OperationType::LessOrEqual, 1);

      if (node->is_sink()) {
        sinks.push_back(node->name);
      }     
    }

    // Add constraints for channels
    // flow_src*coeff_src == flow_dst*coeff_dst
    for (const Chan* channel : graph->chans) {
      const Binding from = channel->source;
      const Binding to = channel->target;

      std::vector<std::string> names{from.node->name, to.node->name};
      std::vector<double> values{from.port->flow, -1.0 * to.port->flow};
      helper->addConstraint(names, values, OperationType::Equal,0);
    }

    // Maximize output flow & solve
    helper->setObjective(sinks, makeCoeffs(sinks));
    helper->setMax();
    helper->solve();
    helper->printProblem();
    helper->printStatus();

    // Print results
    std::vector<double> values = helper->getResults();
    for (double val : values) {
      std::cout<<val<<" ";  
    }
    std::cout<<"\n";
  }
}

double* makeCoeffs(const std::vector<std::string> &sinks) {
  double* sinkCoeffs = new double[sinks.size()];
    for (unsigned int i = 0; i < sinks.size(); i++) {
      sinkCoeffs[i] = 1.0;
    }
  return sinkCoeffs;
}

void LpSolver::checkFlows(const Node* node) {
  if (node->is_merge() || node->is_split()) {
    assert(sumFlows(node->type.inputs) == sumFlows(node->type.outputs));
  }
}

float sumFlows(const std::vector<Argument*> &args) {
  float sum = 0;
  for (const auto* arg : args) {
    sum += arg->flow;
  }
  return sum;
}

}  // namespace eda::hls::scheduler

