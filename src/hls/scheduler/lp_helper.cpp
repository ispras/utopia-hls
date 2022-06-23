//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/scheduler/lp_helper.h"

#include <iostream>
#include <memory>

namespace eda::hls::scheduler {

LpSolverHelper::~LpSolverHelper() {
  deleteFields();
}

void LpSolverHelper::solve() {
  addAllConstraints();

  // Calculate a solution
  status = ::solve(lp);
}

void LpSolverHelper::reset() {
  deleteFields();
  initFields();
}

void LpSolverHelper::deleteFields() {
  for (auto *constraint : constraints) {
    delete constraint;
  }
  for (auto var : variables) {
    delete var.second;
  }
  ::delete_lp(lp);
}

void LpSolverHelper::initFields() {
  lp = ::make_lp(0, 0);
  assert(lp != nullptr && "LP problem creation error!");
  variables = std::map<std::string, SolverVariable*>();
  constraints = std::vector<SolverConstraint*>();
  currentColumn = 0;
  status = -10;
}

std::vector<double> LpSolverHelper::getResults() {
  int size = ::get_Ncolumns(lp);
  double *values = new double[size];
  ::get_variables(lp, values);
  std::vector<double> vec_values;
  vec_values.assign(values, values + size);
  delete [] values;
  return vec_values;
}

SolverConstraint* LpSolverHelper::addConstraint(
    const std::vector<std::string> &names, const std::vector<double> &values, 
    OperationType operation, double rhs) {
  SolverConstraint *constraint = new SolverConstraint(getVariables(names), 
      values, operation, rhs);
  constraints.push_back(constraint);
  //std::cout << "Added constraint: " << *constraint << "\n";
  return constraint;
}

SolverVariable* LpSolverHelper::addVariable(const std::string &name, 
    const model::Node *node) {
  //std::cout << "Adding variable: " << name << "\n";
  SolverVariable *newVariable = 
      new SolverVariable(name, ++currentColumn, node);
  variables[name] = newVariable;
  ::add_column(lp, NULL);
  std::string nameLoc = name;
  ::set_col_name(lp, currentColumn, &nameLoc[0]);
  return newVariable;
}

std::vector<SolverVariable*> LpSolverHelper::getVariables() {
  std::vector<SolverVariable*> result;
  for (const auto &pair : variables) {
    result.push_back(pair.second);
  }
  return result;
}

std::shared_ptr<int[]> getColumnNumbers(
      const std::vector<SolverVariable*> &variables) {
  std::shared_ptr<int[]> colno(new int[variables.size()]);
  int i = 0;
  for (const auto *var : variables) {
    colno[i++] = var->columnNumber;
  }
  return colno;
}

void LpSolverHelper::addAllConstraints() {
  ::set_add_rowmode(lp, TRUE);
  for (const auto *constraint : constraints) {
    std::vector<double> values = constraint->values;
    assert(::add_constraintex(lp, constraint->variables.size(), &values[0], 
        getColumnNumbers(constraint->variables).get(), constraint->operation, 
        constraint->rhs) && "Constraint creation error!");
  }
  ::set_add_rowmode(lp, FALSE);
}

void LpSolverHelper::setObjective(const std::vector<std::string> &names, 
    double *vals) {
  
  ::set_obj_fnex(lp, names.size(), vals, 
      getColumnNumbers(getVariables(names)).get());
}

std::vector<SolverVariable*> LpSolverHelper::getVariables(
    const std::vector<std::string> &names) {
  std::vector<SolverVariable*> vars;
  for (const std::string &name : names) {
    auto it = variables.find(name);
    assert(it != variables.end() && ("Variable " + name + " not found!").c_str());
    vars.push_back(it->second);
  }
  return vars;
}

void LpSolverHelper::setMax() {
  ::set_maxim(lp);
}

void LpSolverHelper::setMin() {
  ::set_minim(lp);
}

int LpSolverHelper::getStatus() {
  return status;
}

void LpSolverHelper::printResults() {
  std::cout << "Solution results:" << std::endl;
  for (auto val : getResults()) {
    std::cout << val << " ";  
  }
  std::cout << std::endl;
}

void LpSolverHelper::printStatus() {
  // Values from lp_lib.h
  switch (status) {
    case NOMEMORY:
      std::cout << "Out of memory" << std::endl;
      break;
    
    case OPTIMAL:
      std::cout << "An optimal solution was obtained" << std::endl;
      break;

    case SUBOPTIMAL:
      std::cout << "A sub-optimal solution was obtained" << std::endl;
      break;

    case INFEASIBLE:
      std::cout << "The model is infeasible" << std::endl;
      break;

    case UNBOUNDED:
      std::cout << "The model is unbounded" << std::endl;
      break;

    case DEGENERATE:
      std::cout << "The model is degenerative" << std::endl;
      break;

    case NUMFAILURE:
      std::cout << "Numerical failure encountered" << std::endl;
      break;

    case USERABORT:
      std::cout << "The abort routine returned TRUE" << std::endl;
      break;

    case TIMEOUT:
      std::cout << "A timeout occurred" << std::endl;
      break;

    case PRESOLVED:
      std::cout << "The model could be solved by presolve" << std::endl;
      break;

    default:
      std::cout<<"Unexpected" << std::endl;
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

  if (constraint.operation == LessOrEqual) {
    out << " <= ";
  } else if (constraint.operation == GreaterOrEqual) {
    out << " >= ";
  } else if (constraint.operation == Equal) {
    out << " = ";
  }
  
  return out << constraint.rhs << std::endl;
}

/*std::ostream& operator <<(std::ostream &out, const LpSolverHelper &problem) {
  

}*/

} // namespace eda::hls::scheduler
