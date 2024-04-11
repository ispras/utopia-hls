#ifndef DFCIR_LP_UTILS_H
#define DFCIR_LP_UTILS_H

#include "DFCIRPassesUtils.h"
#include "lpsolve/lp_lib.h"

#include <unordered_set>


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

    struct LPVariable final {
    public:
        explicit LPVariable(int id);
        LPVariable(const LPVariable &) = default;
        ~LPVariable() = default;
        bool operator== (const LPVariable &other) const;

        int id;
    };

    struct LPConstraint final {
    public:
        LPConstraint(int id, size_t count, int *vars, double *coeffs, OpType op, double rhs);
        LPConstraint(const LPConstraint &other);
        ~LPConstraint();
        bool operator== (const LPConstraint &other) const;

        int id;
        size_t count;
        int *vars;
        double *coeffs;
        OpType op;
        double rhs;
    };

} // namespace mlir::dfcir::utils::lp

template<>
struct std::hash<mlir::dfcir::utils::lp::LPVariable> {
    size_t operator() (const mlir::dfcir::utils::lp::LPVariable &var) const noexcept {
        return var.id;
    }
};

template<>
struct std::hash<mlir::dfcir::utils::lp::LPConstraint> {
    size_t operator() (const mlir::dfcir::utils::lp::LPConstraint &cons) const noexcept {
//        size_t aggr = 0;
//        for (size_t index = 0; index < cons.count; ++index) {
//            aggr += cons.vars[index] * cons.coeffs[index];
//        }
//
//        return aggr + 13 + cons.op + cons.rhs;
        // TODO: Fix in the future.
        return cons.id;
    }
};

namespace mlir::dfcir::utils::lp {
    class LPProblem final {
    public:
        explicit LPProblem();
        ~LPProblem();
        int addVariable();
        void addConstraint(size_t count, int *vars, double *coeffs, OpType op, double rhs);
        int solve();
        void setMin();
        void setMax();
        void setObjective(size_t count, int *vars, double *coeffs);
        int getResults(double **result);
        void lessMessages();

    private:
        void finalizeInit();
        int _current_col;
        // TODO: Change in the future.
        int _current_con;
        lprec *_lp;
        std::unordered_set<LPVariable> variables;
        std::unordered_set<LPConstraint> constraints;
    };

} // namespace mlir::dfcir::utils::lp

#endif // DFCIR_LP_UTILS_H
