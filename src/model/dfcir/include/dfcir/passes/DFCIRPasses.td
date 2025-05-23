//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_Passes
#define DFCIR_Passes

include "mlir/Pass/PassBase.td"

def DFCIRToFIRRTLPass: Pass<"dfcir-to-firrtl-pass", "mlir::ModuleOp"> {
  let summary = "Lower DFCIR entities to FIRRTL entities.";

  let constructor = "mlir::dfcir::createDFCIRToFIRRTLPass()";
}

def DFCIRCombPipelinePass: Pass<"dfcir-comb-pipeline-pass", "mlir::ModuleOp"> {
  let summary = "Divides a combinational design into K pipeline stages.";

  let constructor = "mlir::dfcir::createDFCIRCombPipelinePass()";

  let options = [
    Option<"stages",
           "stages",
           "uint64_t",
           "1",
           "The number of pipeline stages to divide the design into.">
  ];
}

class DFCIRSchedulerPass<string name, string op>: Pass<name, op> {

  let options = [
    Option<"latencyConfig",
           "latencyConfig",
           "LatencyConfig *",
           "nullptr",
           "Latency configuration for operations.">
  ];

  let statistics = [
    Statistic<"latencyStatistic", "latency-statistic", "Top-level kernel overall latency">
  ];
}

def DFCIRASAPSchedulerPass: DFCIRSchedulerPass<"dfcir-asap-scheduler-pass", "mlir::ModuleOp"> {
  let summary = "Schedule operations with a greedy as-soon-as-possible algorithm.";

  let constructor = "mlir::dfcir::createDFCIRASAPSchedulerPass()";
}

def DFCIRLinearSchedulerPass: DFCIRSchedulerPass<"dfcir-linear-scheduler-pass", "mlir::ModuleOp"> {
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
