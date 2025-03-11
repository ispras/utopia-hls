//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/DFCIRDialect.h"
#include "dfcir/DFCIRDialect.cpp.inc"

void mlir::dfcir::DFCIRDialect::initialize() {
  registerOperations();
  registerTypes();
}
