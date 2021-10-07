//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/scheduler/lp_helper.h>

namespace eda::hls::scheduler {

int LpSolverHelper::solve() {
  addAllConstraints();

  // Only important messages on screen while solving
  set_verbose(lp, IMPORTANT);

  // Calculate a solution
  return ::solve(lp);
}

void LpSolverHelper::addConstraint(std::vector<std::string> names, 
    std::vector<double> values, OperationType operation, double rhs) {
  std::vector<SolverVariable*> vars;
  for (const std::string &name : names) {
    auto it = variables.find(name);
    if (it != variables.end()) {
      vars.push_back(it->second);
    } else {
      // TODO: report error
    }
  }

  constraints.push_back(new SolverConstraint(vars, values, 
      static_cast<int>(operation), rhs));
}

/// Constructs and adds a variable to the problem
///
/// \param name variable name
/// \param node corresponding graph node
void LpSolverHelper::addVariable(const std::string &name, Node* node) {
  SolverVariable* new_variable = 
      new SolverVariable(name, ++current_column, node);
  variables[name] = new_variable;
  add_column(lp, NULL);
  std::string name_loc = name;
  set_col_name(lp, current_column, &name_loc[0]);
}

/// Get the existing variables
std::vector<SolverVariable*> LpSolverHelper::getVariables() {
  std::vector<SolverVariable*> result;
  for (const auto &pair : variables) {
    result.push_back(pair.second);
  }
  return result;
}

/// Adds all existing constraints to the problem.
void LpSolverHelper::addAllConstraints() {
  set_add_rowmode(lp, TRUE);
  for (const auto &constraint : constraints) {
    int exprs = constraint->variables.size();
    int* colno = new int[exprs];
    int i = 0;
    for (const auto &var : constraint->variables) {
      colno[i++] = var->column_number;
    }
    add_constraintex(lp, exprs, &(constraint->values)[0], colno, 
        constraint->operation, constraint->rhs);
  }
  set_add_rowmode(lp, FALSE);
}

} // namespace eda::hls::scheduler
