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
// HIL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HIL_HILDIALECT_H
#define HIL_HILDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir::hil {

struct Flow {
  double value;

  Flow(double v) : value(v) {}

  Flow(const Flow &) = default;
  Flow &operator=(const Flow &) = default;

  operator double() const {
    return value;
  }
};

} // namespace mlir::hil

#include "HIL/OpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "HIL/OpsTypes.h.inc"
#define GET_ATTRDEF_CLASSES
#include "HIL/OpsAttributes.h.inc"

#endif // HIL_HILDIALECT_H
