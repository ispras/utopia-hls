## SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS DFCIROpInterfaces.td)
mlir_tablegen(DFCIROpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(DFCIROpInterfaces.cpp.inc -gen-op-interface-defs)
set(LLVM_TARGET_DEFINITIONS DFCIRTypes.td)
mlir_tablegen(DFCIRTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(DFCIRTypeInterfaces.cpp.inc -gen-type-interface-defs)
set(LLVM_TARGET_DEFINITIONS DFCIROperations.td)
mlir_tablegen(DFCIRDialect.h.inc -gen-dialect-decls -dialect=dfcir)
mlir_tablegen(DFCIRDialect.cpp.inc -gen-dialect-defs -dialect=dfcir)
mlir_tablegen(DFCIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(DFCIRTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(DFCIROperations.h.inc -gen-op-decls)
mlir_tablegen(DFCIROperations.cpp.inc -gen-op-defs)

add_public_tablegen_target(DFCIRIncGen)
add_dependencies(mlir-headers DFCIRIncGen)

add_subdirectory(passes)
