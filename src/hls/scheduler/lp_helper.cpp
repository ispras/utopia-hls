//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/scheduler/lp_helper.h>

namespace eda::hls::scheduler {

void LpSolverHelper::solve() {
  addAllConstraints();

  // Only important messages on screen while solving
  set_verbose(lp, IMPORTANT);

  // Calculate a solution
  status = ::solve(lp);
}

void LpSolverHelper::addConstraint(std::vector<std::string> names, 
    std::vector<double> values, OperationType operation, double rhs) {
  std::vector<SolverVariable*> vars;
  std::cout<<"Adding constraint for: ";
  for (const std::string &name : names) {
    std::cout<<name<<" ";
  }
  std::cout<<"\n";
  for (const std::string &name : names) {
    auto it = variables.find(name);
    assert(it != variables.end());
    vars.push_back(it->second);
  }

  constraints.push_back(new SolverConstraint(vars, values, 
      static_cast<int>(operation), rhs));
}

void LpSolverHelper::addVariable(const std::string &name, Node* node) {
  std::cout<<"Adding variable: "<<name<<"\n";
  SolverVariable* new_variable = 
      new SolverVariable(name, ++current_column, node);
  variables[name] = new_variable;
  add_column(lp, NULL);
  std::string name_loc = name;
  set_col_name(lp, current_column, &name_loc[0]);
}

std::vector<SolverVariable*> LpSolverHelper::getVariables() {
  std::vector<SolverVariable*> result;
  for (const auto &pair : variables) {
    result.push_back(pair.second);
  }
  return result;
}

void LpSolverHelper::addAllConstraints() {
  set_add_rowmode(lp, TRUE);
  for (const auto* constraint : constraints) {
    int exprs = constraint->variables.size();
    int* colno = new int[exprs];
    int i = 0;
    for (const auto* var : constraint->variables) {
      colno[i++] = var->column_number;
    }
    std::vector<double> values = constraint->values;
    add_constraintex(lp, exprs, &values[0], colno, 
        constraint->operation, constraint->rhs);
  }
  set_add_rowmode(lp, FALSE);
}

int LpSolverHelper::getStatus() {
  return status;
}

void LpSolverHelper::printStatus() {
  switch (status) {
    case -2:
      std::cout<<"Out of memory\n";
      break;
    
    case 0:
      std::cout<<"An optimal solution was obtained\n";
      break;

    case 1:
      std::cout<<"A sub-optimal solution was obtained\n";
      break;

    case 2:
      std::cout<<"The model is infeasible\n";
      break;

    case 3:
      std::cout<<"The model is unbounded\n";
      break;

    case 4:
      std::cout<<"The model is degenerative\n";
      break;

    case 5:
      std::cout<<"Numerical failure encountered\n";
      break;

    case 6:
      std::cout<<"The abort routine returned TRUE\n";
      break;

    case 7:
      std::cout<<"A timeout occurred\n";
      break;

    case 9:
      std::cout<<"The model could be solved by presolve\n";
      break;

    case 25:
      std::cout<<"Accuracy error encountered\n";
      break;

    default:
      std::cout<<"Unexpected\n";
      break;
    }
}

} // namespace eda::hls::scheduler
