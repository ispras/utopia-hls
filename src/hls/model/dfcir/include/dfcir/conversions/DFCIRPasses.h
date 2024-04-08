#ifndef DFCIR_PASSES_H
#define DFCIR_PASSES_H

#include "dfcir/DFCIROperations.h"
#include "memory"
#include "mlir/Pass/Pass.h"

namespace mlir::dfcir {

    enum Ops {
        UNDEFINED,
        ADD_INT,
        ADD_FLOAT,
        MUL_INT,
        MUL_FLOAT
    };
} // namespace mlir::dfcir

typedef std::unordered_map<mlir::dfcir::Ops, unsigned> LatencyConfig;


namespace mlir::dfcir {
    std::unique_ptr<mlir::Pass> createDFCIRToFIRRTLPass(LatencyConfig *config = nullptr);
    std::unique_ptr<mlir::Pass> createDFCIRDijkstraSchedulerPass();
    std::unique_ptr<mlir::Pass> createDFCIRLinearSchedulerPass();
} // namespace mlir::dfcir

#define GEN_PASS_REGISTRATION
#include "dfcir/conversions/DFCIRPasses.h.inc"

#endif // DFCIR_PASSES_H
