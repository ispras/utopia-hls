set(LLVM_TARGET_DEFINITIONS DFCIRPasses.td)
mlir_tablegen(DFCIRPasses.h.inc --gen-pass-decls)

add_public_tablegen_target(DFCIRPassesIncGen)
add_dependencies(DFCIRPassesIncGen DFCIRIncGen)
