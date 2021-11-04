//===- HILCombine.h - HIL dialect -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HIL_HILCOMBINE_H
#define HIL_HILCOMBINE_H

#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createGraphRewritePass();

#endif // HIL_HILCOMBINE_H
