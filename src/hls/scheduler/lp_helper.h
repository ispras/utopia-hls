//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the declaration of the LpSolverHelper class and its
/// supplement structures, that should be used for lp_solver problem building 
/// and invocation.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <hls/model/model.h>
#include <lpsolve/lp_lib.h>
#include <map>
#include <string>
#include <vector>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

enum OperationType {
  LessOrEqual = 1,
  GreaterOrEqual = 2,
  Equal = 3
};

struct SolverVariable;
struct SolverConstraint;

class LpSolverHelper final {

public:
  LpSolverHelper() : current_column(0), status(-10) {
    lp = make_lp(0, 0);
    set_maxim(lp);
  }

  /// Solves the formulated problem.
  void solve(); 

  void getResults();

  /// Constructs a constraint
  /// 
  /// \param names variable names
  /// \param values variable coefficients
  /// \param operation operation
  /// \param rhs right-hand side value
  void addConstraint(std::vector<std::string> names, 
    std::vector<double> values, OperationType operation, double rhs);

  /// Constructs and adds a variable to the problem
  ///
  /// \param name variable name
  /// \param node corresponding graph node
  void addVariable(const std::string&, Node*);

  /// Prints the problem
  void printProblem() { write_LP(lp, stdout); }

  /// Prints the last solution status
  void printStatus();

  int getStatus();

  /// Get the existing variables
  std::vector<SolverVariable*> getVariables();

  /// Get the existing constraints
  std::vector<SolverConstraint*> getConstraints() { return constraints; }

private:
  /// Adds all existing constraints to the problem.
  void addAllConstraints();

  lprec* lp;
  std::map<std::string, SolverVariable*> variables;
  std::vector<SolverConstraint*> constraints;
  int current_column;
  int status;
};

struct SolverVariable final {

  SolverVariable(const std::string &name, int column_number, Node* node) :
    name(name), column_number(column_number), node(node) {}

  std::string name;
  int column_number;
  Node* node;
};

struct SolverConstraint final {

  SolverConstraint(std::vector<SolverVariable*> variables, 
      std::vector<double> values, int operation, double rhs) : 
      variables(variables), values(values), operation(operation), rhs(rhs) {
    
    assert(variables.size() == values.size());
    assert((operation == 1) || (operation == 2) || (operation == 3));
  }

  std::vector<SolverVariable*> variables;
  std::vector<double> values;
  int operation;
  double rhs;

};

std::ostream& operator <<(std::ostream &out, const LpSolverHelper &problem);
std::ostream& operator <<(std::ostream &out, 
    const SolverConstraint &constraint);

} // namespace eda::hls::scheduler
