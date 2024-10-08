//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_Passes
#define DFCIR_Passes

include "mlir/Pass/PassBase.td"

def DFCIRToFIRRTLPass: Pass<"dfcir-to-firrtl-pass", "mlir::ModuleOp"> {
  let summary = "Lower DFCIR entities to FIRRTL entities.";

  let options = [
    Option<"latencyConfig",
           "latencyConfig",
           "LatencyConfig *",
           "nullptr",
           "Latency configuration for operations.">
  ];

  let constructor = "mlir::dfcir::createDFCIRToFIRRTLPass()";
}

def DFCIRASAPSchedulerPass: Pass<"dfcir-asap-scheduler-pass", "mlir::ModuleOp"> {
  let summary = "Schedule operations with a greedy as-soon-as-possible algorithm.";

  let constructor = "mlir::dfcir::createDFCIRASAPSchedulerPass()";
}

def DFCIRLinearSchedulerPass: Pass<"dfcir-linear-scheduler-pass", "mlir::ModuleOp"> {
  let summary = "Schedule operations with linear programming algorithm.";

  let constructor = "mlir::dfcir::createDFCIRLinearSchedulerPass()";
}

def FIRRTLStubGeneratorPass: Pass<"firrtl-stub-generator-pass", "mlir::ModuleOp"> {
  let summary = "Generate stub modules for pipelined computational operations.";

  let options = [
    Option<"stream",
           "stream",
           "llvm::raw_ostream *",
           "nullptr",
           "Stream to dump stubs to.">
  ];

  let constructor = "mlir::dfcir::createFIRRTLStubGeneratorPass()";
}

#endif // DFCIR_Passes
