#ifndef DFCIR_OPERATIONS_H
#define DFCIR_OPERATIONS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "DFCIROperations.h.inc"

#endif // DFCIR_OPERATIONS_H
