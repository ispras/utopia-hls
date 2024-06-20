#ifndef DFCIR_TYPES_H
#define DFCIR_TYPES_H

#include "dfcir/DFCIRDialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "dfcir/DFCIRTypeInterfaces.h.inc" // Cannot enforce header sorting.
#define GET_TYPEDEF_CLASSES
#include "dfcir/DFCIRTypes.h.inc" // Cannot enforce header sorting.

#endif // DFCIR_TYPES_H
