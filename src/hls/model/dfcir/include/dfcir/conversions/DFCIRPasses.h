#ifndef DFCIR_PASSES_H
#define DFCIR_PASSES_H

#include "dfcir/DFCIROperations.h"
#include "mlir/Pass/Pass.h"

#include "memory"

namespace mlir::dfcir {

enum Ops {
  UNDEFINED,
  ADD_INT,
  ADD_FLOAT,
  SUB_INT,
  SUB_FLOAT,
  MUL_INT,
  MUL_FLOAT,
  DIV_INT,
  DIV_FLOAT,
  AND_INT,
  AND_FLOAT,
  OR_INT,
  OR_FLOAT,
  XOR_INT,
  XOR_FLOAT,
  NOT_INT,
  NOT_FLOAT,
  NEG_INT,
  NEG_FLOAT,
  LESS_INT,
  LESS_FLOAT,
  LESS_EQ_INT,
  LESS_EQ_FLOAT,
  MORE_INT,
  MORE_FLOAT,
  MORE_EQ_INT,
  MORE_EQ_FLOAT,
  EQ_INT,
  EQ_FLOAT,
  NEQ_INT,
  NEQ_FLOAT
};

} // namespace mlir::dfcir

typedef std::unordered_map<mlir::dfcir::Ops, unsigned> LatencyConfig;


namespace mlir::dfcir {

using std::unique_ptr;
using mlir::Pass;

unique_ptr<Pass> createDFCIRToFIRRTLPass(LatencyConfig *config = nullptr);

unique_ptr<Pass> createDFCIRDijkstraSchedulerPass();

unique_ptr<Pass> createDFCIRLinearSchedulerPass();

} // namespace mlir::dfcir

#define GEN_PASS_REGISTRATION

#include "dfcir/conversions/DFCIRPasses.h.inc"

#endif // DFCIR_PASSES_H
