//===- Conversion.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares pass which will lower HIL dialect to FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HIL_CONVERSION_H
#define HIL_CONVERSION_H

#include "HIL/Model.h"
#include "HIL/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using Pass = mlir::Pass;

std::unique_ptr<Pass> createHILToFIRRTLPass();

#endif // HIL_CONVERSION_H