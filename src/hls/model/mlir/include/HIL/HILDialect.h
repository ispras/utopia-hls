//===- HILDialect.h - HIL dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HIL_HILDIALECT_H
#define HIL_HILDIALECT_H

#include "mlir/IR/Dialect.h"

#include "HIL/HILOpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "HIL/HILOpsTypes.h.inc"

#endif // HIL_HILDIALECT_H
