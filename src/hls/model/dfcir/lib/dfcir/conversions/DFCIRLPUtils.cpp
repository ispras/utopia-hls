#include "dfcir/conversions/DFCIRLPUtils.h"

namespace mlir::dfcir::utils::lp {
    LPVariable::LPVariable(int id) : id(id) {}

    bool LPVariable::operator==(const mlir::dfcir::utils::lp::LPVariable &other) const {
        return id == other.id;
    }

    LPConstraint::LPConstraint(int id, size_t count, int *vars, double *coeffs, OpType op, double rhs): id(id), count(count), vars(vars),coeffs(coeffs), op(op), rhs(rhs) { }

    LPConstraint::LPConstraint(const LPConstraint &other) : id(other.id), count(other.count), op(other.op), rhs(other.rhs) {
        vars = new int[count];
        coeffs = new double[count];

        for (size_t index = 0; index < count; ++index) {
            vars[index] = other.vars[index];
            coeffs[index] = other.coeffs[index];
        }
    }

    LPConstraint::~LPConstraint() {
        delete []vars;
        delete []coeffs;
    }

    bool LPConstraint::operator==(const mlir::dfcir::utils::lp::LPConstraint &other) const {
//        if (count != other.count) {
//            return false;
//        }
//
//        for (size_t index = 0; index < count; ++index) {
//            if (vars[index] != other.vars[index] || coeffs[index] != other.coeffs[index]) {
//                return false;
//            }
//        }
//
//        return op == other.op && rhs == other.rhs;
        // TODO: Fix in the future.
        return id == other.id;
    }

    int LPProblem::addVariable() {
        auto it = variables.emplace(_current_col++);
        assert(it.second);
        return it.first->id;
    }

    void LPProblem::addConstraint(size_t count, int *vars, double *coeffs, OpType op,
                                          double rhs) {
        auto it = constraints.emplace(_current_con++, count, vars, coeffs, op, rhs);
        assert(it.second);
    }

    void LPProblem::finalizeInit() {
        for (const LPVariable &var : variables) {
            ::add_column(_lp, NULL);
        }

        ::set_add_rowmode(_lp, TRUE);

        for (const LPConstraint &cons : constraints) {
            assert(::add_constraintex(_lp, cons.count, cons.coeffs,
                                      cons.vars, cons.op, cons.rhs) &&
                                      "Constraint creation error!");
        }

        ::set_add_rowmode(_lp, FALSE);
    }

    int LPProblem::solve() {
        finalizeInit();

        return ::solve(_lp);
    }

    void LPProblem::setMin() {
        ::set_minim(_lp);
    }
    void LPProblem::setMax() {
        ::set_maxim(_lp);
    }

    void LPProblem::setObjective(size_t count, int *vars, double *coeffs) {
        ::set_obj_fnex(_lp, count, coeffs, vars);
    }

    int LPProblem::getResults(double **result) {
        int size = ::get_Ncolumns(_lp);
        *result = new double[size];
        ::get_variables(_lp, *result);
        return size;
    }

    void LPProblem::lessMessages() {
        ::set_verbose(_lp, Verbosity::Critical);
    }

    LPProblem::LPProblem() : _current_col(1), _current_con(1), _lp(::make_lp(0, 0)) {}
} // namespace mlir::dfcir::utils::lp