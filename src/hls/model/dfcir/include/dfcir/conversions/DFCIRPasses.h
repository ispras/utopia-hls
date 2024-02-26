#ifndef DFCIR_PASSES_H
#define DFCIR_PASSES_H

#include "dfcir/DFCIROperations.h"
#include "memory"
#include "mlir/Pass/Pass.h"

namespace mlir::dfcir {

std::unique_ptr<mlir::Pass> createDFCIRToFIRRTLPass();

} // namespace mlir::dfcir

#define GEN_PASS_REGISTRATION
#include "dfcir/conversions/DFCIRPasses.h.inc"

#endif // DFCIR_PASSES_H
