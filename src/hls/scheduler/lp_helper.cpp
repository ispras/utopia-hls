//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/scheduler/lp_helper.h>
#include <memory>

namespace eda::hls::scheduler {

LpSolverHelper::~LpSolverHelper() {

    for (SolverConstraint* constraint : constraints) {
      delete constraint;
    }

    for (auto var : variables) {
      delete var.second;
    }

    delete_lp(lp);
  }

void LpSolverHelper::solve() {
  addAllConstraints();

  // Calculate a solution
  status = ::solve(lp);
}

std::vector<double> LpSolverHelper::getResults() {
  int size = get_Ncolumns(lp);
  double* values = new double[size];
  get_variables(lp, values);
  std::vector<double> vec_values;
  vec_values.assign(values, values + size);
  delete [] values;
  return vec_values;
}

void LpSolverHelper::addConstraint(const std::vector<std::string> &names, 
    const std::vector<double> &values, OperationType operation, double rhs) {
  SolverConstraint *constraint = new SolverConstraint(findVariables(names), 
      values, operation, rhs);
  constraints.push_back(constraint);
  std::cout<<"Added constraint: "<<*constraint<<"\n";
}

void LpSolverHelper::addVariable(const std::string &name, Node* const node) {
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

std::shared_ptr<int[]> getColumnNumbers(const std::vector<SolverVariable*> &variables) {
  std::shared_ptr<int[]> colno(new int[variables.size()]);
  int i = 0;
  for (auto* const var : variables) {
    colno[i++] = var->column_number;
  }
  return colno;
}

void LpSolverHelper::addAllConstraints() {
  set_add_rowmode(lp, TRUE);
  for (auto* const constraint : constraints) {
    std::vector<double> values = constraint->values;
    assert(add_constraintex(lp, constraint->variables.size(), &values[0], 
        getColumnNumbers(constraint->variables).get(), constraint->operation, 
        constraint->rhs));
  }
  set_add_rowmode(lp, FALSE);
}

void LpSolverHelper::setObjective(const std::vector<std::string> &names, 
    double *vals) {
  
  set_obj_fnex(lp, names.size(), vals, 
      getColumnNumbers(findVariables(names)).get());
}

std::vector<SolverVariable*> LpSolverHelper::findVariables(
    const std::vector<std::string> &names) {
  std::vector<SolverVariable*> vars;
  for (const std::string &name : names) {
    auto it = variables.find(name);
    assert(it != variables.end());
    vars.push_back(it->second);
  }
  return vars;
}

void LpSolverHelper::setMax() {
  set_maxim(lp);
}

void LpSolverHelper::setMin() {
  set_minim(lp);
}

int LpSolverHelper::getStatus() {
  return status;
}

void LpSolverHelper::printResults() {
  std::cout<<"Solution results:\n";
  for (double val : getResults()) {
    std::cout<<val<<" ";  
  }
  std::cout<<"\n";
}

void LpSolverHelper::printStatus() {
  // Values from lp_lib.h
  switch (status) {
    case NOMEMORY:
      std::cout<<"Out of memory\n";
      break;
    
    case OPTIMAL:
      std::cout<<"An optimal solution was obtained\n";
      break;

    case SUBOPTIMAL:
      std::cout<<"A sub-optimal solution was obtained\n";
      break;

    case INFEASIBLE:
      std::cout<<"The model is infeasible\n";
      break;

    case UNBOUNDED:
      std::cout<<"The model is unbounded\n";
      break;

    case DEGENERATE:
      std::cout<<"The model is degenerative\n";
      break;

    case NUMFAILURE:
      std::cout<<"Numerical failure encountered\n";
      break;

    case USERABORT:
      std::cout<<"The abort routine returned TRUE\n";
      break;

    case TIMEOUT:
      std::cout<<"A timeout occurred\n";
      break;

    case PRESOLVED:
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

std::ostream& operator <<(std::ostream &out, 
    const SolverConstraint &constraint) {

  
  for (unsigned int i = 0; i < constraint.variables.size(); i++) {
    if (i != 0) {
      out << " + ";
    }
    out << constraint.values[i] << "*" << constraint.variables[i]->name;
  }

  switch (constraint.operation)
  {
  case 1:
    out << " <= ";
    break;

  case 2:
    out << " >= ";
    break;

  case 3:
    out << " = ";
    break;
  }

  return out << constraint.rhs << "\n";

}

} // namespace eda::hls::scheduler
