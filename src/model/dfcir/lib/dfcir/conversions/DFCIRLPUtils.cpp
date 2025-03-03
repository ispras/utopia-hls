//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/conversions/DFCIRLPUtils.h"

#include <stdlib.h>

namespace mlir::dfcir::utils::lp {

LPConstraint::LPConstraint(size_t count, int *vars, double *coeffs,
                           OpType op, double rhs) : count(count),
                                                    vars(vars), coeffs(coeffs),
                                                    op(op), rhs(rhs) {}

LPConstraint::LPConstraint(const LPConstraint &other) : count(other.count),
                                                        op(other.op),
                                                        rhs(other.rhs) {
  vars = (int *) calloc(count, sizeof(int));
  coeffs = (double *) calloc(count, sizeof(double));

  for (size_t index = 0; index < count; ++index) {
    vars[index] = other.vars[index];
    coeffs[index] = other.coeffs[index];
  }
}

LPConstraint::~LPConstraint() {
  free(vars);
  free(coeffs);
}

int LPProblem::addVariable() {
  return currentCol++;
}

void LPProblem::addConstraint(size_t count, int *vars,
                              double *coeffs, OpType op, double rhs) {
  constraints.emplace_back(count, vars, coeffs, op, rhs);
}

void LPProblem::finalizeInit() {
  assert(::resize_lp(lp, constraints.size(), currentCol - 1));

  for (int i = 0; i < (currentCol - 1); ++i) {
    // Issue #8 (https://github.com/ispras/utopia-hls/issues/8).
    assert(::add_column(lp, NULL));
  }

  ::set_add_rowmode(lp, TRUE);

  for (const LPConstraint &cons: constraints) {
    auto successful = ::add_constraintex(lp, cons.count, cons.coeffs,
                                         cons.vars, cons.op, cons.rhs);
    assert(successful && "Constraint creation error!");
  }

  ::set_add_rowmode(lp, FALSE);
}

int LPProblem::solve() {
  finalizeInit();

  return ::solve(lp);
}

void LPProblem::setMin() {
  ::set_minim(lp);
}

void LPProblem::setMax() {
  ::set_maxim(lp);
}

void LPProblem::setObjective(size_t count, int *vars, double *coeffs) {
  ::set_obj_fnex(lp, count, coeffs, vars);
}

int LPProblem::getResults(double **result) {
  int size = ::get_Ncolumns(lp);
  *result = (double *) calloc(size, sizeof(double));
  ::get_variables(lp, *result);
  return size;
}

void LPProblem::lessMessages() {
  ::set_verbose(lp, Verbosity::Critical);
}

LPProblem::LPProblem() : currentCol(1), lp(::make_lp(0, 0)) {}

LPProblem::~LPProblem() {
  delete_lp(lp);
}

} // namespace mlir::dfcir::utils::lp
