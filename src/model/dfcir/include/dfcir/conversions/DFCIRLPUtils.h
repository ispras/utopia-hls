//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_LP_UTILS_H
#define DFCIR_LP_UTILS_H

#include "DFCIRPassesUtils.h"

#include "lpsolve/lp_lib.h"

#include <vector>

namespace mlir::dfcir::utils::lp {

// Values from lp_lib.h .
enum OpType {
  LessOrEqual = LE,
  GreaterOrEqual = GE,
  Equal = EQ
};

// Values from lp_lib.h .
enum Status {
  NoMemory = NOMEMORY,
  Optimal = OPTIMAL,
  Suboptimal = SUBOPTIMAL,
  Infeasible = INFEASIBLE,
  Unbounded = UNBOUNDED,
  Degenerate = DEGENERATE,
  NumFailure = NUMFAILURE,
  UserAbort = USERABORT,
  Timeout = TIMEOUT,
  Presolved = PRESOLVED
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

struct LPConstraint final {
public:
  size_t count;
  int *vars;
  double *coeffs;
  OpType op;
  double rhs;

  LPConstraint(size_t count, int *vars,
               double *coeffs, OpType op, double rhs);

  LPConstraint(const LPConstraint &other);

  ~LPConstraint();
};

} // namespace mlir::dfcir::utils::lp

namespace mlir::dfcir::utils::lp {

class LPProblem final {
private:
  int currentCol;
  lprec *lp;
  std::vector<LPConstraint> constraints;

  void finalizeInit();

public:
  explicit LPProblem();

  ~LPProblem();

  int addVariable();

  void addConstraint(size_t count, int *vars, double *coeffs,
                     OpType op, double rhs);

  int solve();

  void setMin();

  void setMax();

  void setObjective(size_t count, int *vars, double *coeffs);

  int getResults(double **result);

  void lessMessages();
};

} // namespace mlir::dfcir::utils::lp

#endif // DFCIR_LP_UTILS_H
