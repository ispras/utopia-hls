//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
// Graph rewrite pass.
//
//===----------------------------------------------------------------------===//

#ifndef HIL_HILCOMBINE_H
#define HIL_HILCOMBINE_H

#include "HIL/Model.h"

#include "mlir/Pass/Pass.h"

using Pass = mlir::Pass;

std::unique_ptr<Pass> createGraphRewritePass();

#endif // HIL_HILCOMBINE_H