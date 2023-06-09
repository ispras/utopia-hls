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
// HIL dialect ops.
//
//===----------------------------------------------------------------------===//

#include "HIL/Ops.h"

#include "HIL/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "HIL/Ops.cpp.inc"