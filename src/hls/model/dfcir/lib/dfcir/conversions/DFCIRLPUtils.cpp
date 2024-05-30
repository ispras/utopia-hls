#include "dfcir/conversions/DFCIRLPUtils.h"

namespace mlir::dfcir::utils::lp {

LPVariable::LPVariable(int id) : id(id) {}

bool
LPVariable::operator==(const mlir::dfcir::utils::lp::LPVariable &other) const {
  return id == other.id;
}

LPConstraint::LPConstraint(int id, size_t count, int *vars, double *coeffs,
                           OpType op, double rhs) : id(id), count(count),
                                                    vars(vars), coeffs(coeffs),
                                                    op(op), rhs(rhs) {}

LPConstraint::LPConstraint(const LPConstraint &other) : id(other.id),
                                                        count(other.count),
                                                        op(other.op),
                                                        rhs(other.rhs) {
  vars = new int[count];
  coeffs = new double[count];

  for (size_t index = 0; index < count; ++index) {
    vars[index] = other.vars[index];
    coeffs[index] = other.coeffs[index];
  }
}

LPConstraint::~LPConstraint() {
  delete[]vars;
  delete[]coeffs;
}

bool LPConstraint::operator==(
        const mlir::dfcir::utils::lp::LPConstraint &other) const {
  // TODO: Fix in the future.
  return id == other.id;
}

int LPProblem::addVariable() {
  auto it = variables.emplace(currentCol++);
  assert(it.second);
  return it.first->id;
}

void
LPProblem::addConstraint(size_t count, int *vars, double *coeffs, OpType op,
                         double rhs) {
  auto it = constraints.emplace(currentCon++, count, vars, coeffs, op, rhs);
  assert(it.second);
}

void LPProblem::finalizeInit() {
  for (const LPVariable &var: variables) {
    ::add_column(lp, NULL);
  }

  ::set_add_rowmode(lp, TRUE);

  for (const LPConstraint &cons: constraints) {
    unsigned char successful = ::add_constraintex(lp, cons.count, cons.coeffs,
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
  *result = new double[size];
  ::get_variables(lp, *result);
  return size;
}

void LPProblem::lessMessages() {
  ::set_verbose(lp, Verbosity::Critical);
}

LPProblem::LPProblem() : currentCol(1), currentCon(1), lp(::make_lp(0, 0)) {}

LPProblem::~LPProblem() {
  delete_lp(lp);
}

} // namespace mlir::dfcir::utils::lp