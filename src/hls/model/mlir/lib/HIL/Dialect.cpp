//===- HILDialect.cpp - HIL dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIL/Dialect.h"
#include "HIL/Ops.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hil;

#include "HIL/OpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "HIL/OpsTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

//===----------------------------------------------------------------------===//
// HIL dialect.
//===----------------------------------------------------------------------===//
void HILDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HIL/Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "HIL/OpsTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
      >();
}

Type HILDialect::parseType(DialectAsmParser &parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type))
    return Type();
  mlir::Type value;
  generatedTypeParser(parser, data_type, value);
  return value;
}

void HILDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}
