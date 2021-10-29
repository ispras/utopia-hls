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

// Values from lp_lib.h
enum OperationType {
  LessOrEqual = LE,
  GreaterOrEqual = GE,
  Equal = EQ
};

// Values from lp_lib.h
enum Verbosity {
  Neutral = NEUTRAL,
  Critical = CRITICAL,
  Severe = SEVERE,
  Important = IMPORTANT,
  Normal = NORMAL,
  Detailed = DETAILED,
  Full = FULL
};

struct SolverVariable;
struct SolverConstraint;

class LpSolverHelper final {

public:

  ~LpSolverHelper();

  /// Solves the formulated problem.
  void solve(); 

  /// Returns the solution results.
  std::vector<double> getResults();

  /// Constructs a constraint.
  /// 
  /// \param names variable names
  /// \param values variable coefficients
  /// \param operation operation
  /// \param rhs right-hand side value
  SolverConstraint* addConstraint(const std::vector<std::string> &names, 
    const std::vector<double> &values, OperationType operation, double rhs);

  /// Constructs and adds a variable to the problem.
  ///
  /// \param name variable name
  /// \param node corresponding graph node
  SolverVariable* addVariable(const std::string&, const Node*);

  /// Prints the problem.
  void printProblem() { write_LP(lp, stdout); }

  /// Prints the last solution status.
  void printStatus();

  void printResults();

  /// Returns the solution status.
  int getStatus();

  /// Sets the optimization objective.
  void setObjective(const std::vector<std::string> &names, double *vals);

  /// Maximizes the solution.
  void setMax();

  /// Minimizes the solution.
  void setMin();

  /// Sets the output verbosity.
  void setVerbosity(Verbosity verb) { set_verbose(lp, verb); }

  /// Get the existing variables.
  std::vector<SolverVariable*> getVariables();

  /// Searches for the variables with the given names.
  std::vector<SolverVariable*> 
      getVariables(const std::vector<std::string> &names);

  /// Get the existing constraints.
  std::vector<SolverConstraint*> getConstraints() { return constraints; }

  static LpSolverHelper* get();
  static LpSolverHelper* reset();

private:
  LpSolverHelper() : lp(make_lp(0, 0)), currentColumn(0), status(-10) {
    setVerbosity(Normal);
  }

  LpSolverHelper(LpSolverHelper &other) = delete;
  void operator=(const LpSolverHelper &other) = delete;

  /// Adds all existing constraints to the problem.
  void addAllConstraints();

  static LpSolverHelper* instance;
  lprec* lp;
  std::map<std::string, SolverVariable*> variables;
  std::vector<SolverConstraint*> constraints;
  unsigned currentColumn;
  int status;
};

struct SolverVariable final {

  SolverVariable(const std::string &name, unsigned column_number, 
      const Node* node) : name(name), column_number(column_number), 
      node(node) { }

  std::string name;
  unsigned column_number;
  const Node* node;
};

struct SolverConstraint final {

  SolverConstraint(const std::vector<SolverVariable*> &variables, 
      const std::vector<double> &values, OperationType operation, double rhs) : 
      variables(variables), values(values), operation(operation), rhs(rhs) {
    
    assert(variables.size() == values.size());
  }

  std::vector<SolverVariable*> variables;
  std::vector<double> values;
  OperationType operation;
  double rhs;

};

std::ostream& operator <<(std::ostream &out, const LpSolverHelper &problem);
std::ostream& operator <<(std::ostream &out, 
    const SolverConstraint &constraint);

} // namespace eda::hls::scheduler
