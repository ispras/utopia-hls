//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_OPERATIONS_H
#define DFCIR_OPERATIONS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "dfcir/DFCIRDialect.h" // Cannot enforce header sorting.
#include "dfcir/DFCIRTypes.h"
#include "dfcir/DFCIROpInterfaces.h"
#define GET_OP_CLASSES
#include "dfcir/DFCIROperations.h.inc"

#endif // DFCIR_OPERATIONS_H
