//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===

#include <cassert>
#include <hls/scheduler/lp_helper.h>
#include <hls/scheduler/scheduler.h>
#include <memory>

namespace eda::hls::scheduler {

void LpSolver::setModel(const std::string&) {

}

void LpSolver::balance() {

  for (const Graph* graph : model->graphs) {
    LpSolverHelper helper;
    for (Node* node : graph->nodes) {
      checkFlows(node);
      std::string nodeName = node->type.name;
      helper.addVariable(nodeName, node);

      std::vector<std::string> names{nodeName};
      std::vector<double> valOne{1.0};

      // Add 'coeff > 0'
      helper.addConstraint(names, valOne, OperationType::GreaterOrEqual, 0);
      // Add 'coeff < 1'
      helper.addConstraint(names, valOne, OperationType::LessOrEqual, 1);     
    }

    for (const Chan* channel : graph->chans) {
      const Argument* from = channel->source;
      const Argument* to = channel->target;

      std::vector<std::string> names{from->name, to->name};
      std::vector<double> values{from->flow, -1.0 * to->flow};
      helper.addConstraint(names, values, OperationType::Equal,0);
    }

    helper.printProblem();
    helper.solve();
    helper.printStatus();

  }
}

void LpSolver::checkFlows(const Node* node) {
  int isSpecial = isMergeSplit(node->type.name);
  if (isSpecial != 0) {
    float inputFlow = sumFlows(node->type.inputs);
    float outputFlow = sumFlows(node->type.outputs);
    assert(inputFlow == outputFlow);
  }
}

float sumFlows(std::vector<Argument*> args) {
  float sum = 0;
  for (const auto* arg : args) {
    sum += arg->flow;
  }
  return sum;
}

int isMergeSplit(const std::string &nodeName) {
  return checkMerge(nodeName) ? 1 : checkSplit(nodeName) ? 2 : 0;
}

bool checkMerge(const std::string &nodeName) {
  return nodeName.compare(MERGE) == 0;
}

bool checkSplit(const std::string &nodeName) {
  return nodeName.compare(SPLIT) == 0;
}

}  // namespace eda::hls::scheduler

