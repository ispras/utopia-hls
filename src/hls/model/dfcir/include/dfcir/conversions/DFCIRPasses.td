#ifndef DFCIR_Passes
#define DFCIR_Passes

include "mlir/Pass/PassBase.td"

def DFCIRToFIRRTLPass: Pass<"dfcir-to-firrtl-pass", "::mlir::ModuleOp"> {
	let summary = "Lower DFCIR entities to FIRRTL entities.";
	let constructor = "::mlir::dfcir::createDFCIRToFIRRTLPass()";
}

#endif // DFCIR_Passes