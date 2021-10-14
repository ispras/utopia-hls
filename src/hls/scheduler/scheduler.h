//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <hls/model/model.h>
#include <hls/scheduler/lp_helper.h>
#include <string>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class LpSolver final {

public:

  LpSolver(Model* model_arg) : model(model_arg) { 
    helper = new LpSolverHelper;
  }

  LpSolver() { helper = new LpSolverHelper; }

  void setModel(Model* model_arg) { model = model_arg; }

  void balance();

  int getResult();

private:

  void checkFlows(const Node* node);
  
  Model* model;
  LpSolverHelper* helper;

};

float sumFlows(std::vector<Argument*> args);

} // namespace eda::hls::scheduler
