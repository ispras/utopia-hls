//===- Conversion.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower HIL dialect to
// FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HIL_CONVERSION_H
#define HIL_CONVERSION_H

#include "HIL/Ops.h"

#include "mlir/IR/Builders.h"

namespace mlir {
class Pass;
} // namespace mlir

std::unique_ptr<mlir::Pass> createHILToFIRRTLPass();

namespace mlir::hil {

} // namespace mlir::hil

#endif // HIL_CONVERSION_H