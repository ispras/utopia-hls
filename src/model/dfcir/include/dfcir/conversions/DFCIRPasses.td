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

def DFCIRDijkstraSchedulerPass: Pass<"dfcir-dijkstra-scheduler-pass", "::mlir::ModuleOp"> {
    let summary = "Schedule operations with Dijkstra algorithm.";
    //let options = [
    //    Option<"latencyConfig",
    //           "latencyConfig",
    //           "LatencyConfig *",
    //           "nullptr",
    //           "Latency configuration for operations to schedule.">
    //];
    let constructor = "::mlir::dfcir::createDFCIRDijkstraSchedulerPass()";
}

def DFCIRLinearSchedulerPass: Pass<"dfcir-linear-scheduler-pass", "::mlir::ModuleOp"> {
    let summary = "Schedule operations with linear programming algorithm.";
    //let options = [
    //    Option<"latencyConfig",
    //           "latencyConfig",
    //           "LatencyConfig *",
    //           "nullptr",
    //           "Latency configuration for operations to schedule.">
    //];
    let constructor = "::mlir::dfcir::createDFCIRLinearSchedulerPass()";
}

#endif // DFCIR_Passes