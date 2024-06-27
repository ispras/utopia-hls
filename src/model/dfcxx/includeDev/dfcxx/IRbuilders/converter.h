//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_IR_BUILDER_CONVERTER_H
#define DFCXX_IR_BUILDER_CONVERTER_H

#include "dfcir/DFCIROperations.h"
#include "dfcxx/kernel.h"
#include "dfcxx/typedefs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace dfcxx {

class DFCIRTypeConverter {
  mlir::MLIRContext *ctx;

public:
  DFCIRTypeConverter(mlir::MLIRContext *ctx);

  mlir::Type operator[](DFVariableImpl *var);
};

} // namespace dfcxx

#endif // DFCXX_IR_BUILDER_CONVERTER_H
