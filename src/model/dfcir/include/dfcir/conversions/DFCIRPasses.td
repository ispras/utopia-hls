#ifndef DFCIR_Passes
#define DFCIR_Passes

include "mlir/Pass/PassBase.td"

def DFCIRToFIRRTLPass: Pass<"dfcir-to-firrtl-pass", "::mlir::ModuleOp"> {
  let summary = "Lower DFCIR entities to FIRRTL entities.";
  let options = [
    Option<"latencyConfig",
           "latencyConfig",
           "LatencyConfig *",
           "nullptr",
           "Latency configuration for operations.">
  ];
  let constructor = "::mlir::dfcir::createDFCIRToFIRRTLPass()";
}

def DFCIRASAPSchedulerPass: Pass<"dfcir-asap-scheduler-pass", "::mlir::ModuleOp"> {
  let summary = "Schedule operations with a greedy as-soon-as-possible algorithm.";
  let constructor = "::mlir::dfcir::createDFCIRASAPSchedulerPass()";
}

def DFCIRLinearSchedulerPass: Pass<"dfcir-linear-scheduler-pass", "::mlir::ModuleOp"> {
  let summary = "Schedule operations with linear programming algorithm.";
  let constructor = "::mlir::dfcir::createDFCIRLinearSchedulerPass()";
}

#endif // DFCIR_Passes
