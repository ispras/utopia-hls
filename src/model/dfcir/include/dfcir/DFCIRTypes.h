//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

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
