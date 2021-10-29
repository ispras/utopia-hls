//===- HILDialect.cpp - HIL dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HIL/HILDialect.h"
#include "HIL/HILOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hil;

#include "HIL/HILOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "HIL/HILOpsTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

//===----------------------------------------------------------------------===//
// HIL dialect.
//===----------------------------------------------------------------------===//
void HILDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HIL/HILOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "HIL/HILOpsTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
    >();
}

Type HILDialect::parseType(DialectAsmParser &parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();
  mlir::Type value;
  generatedTypeParser(parser, data_type, value);
  return value;
}
  //return HILLolType::parse(parser);
/*   // If this dialect allows unknown types, then represent this with OpaqueType. */
/*   if (allowsUnknownTypes()) { */
/*     Identifier ns = Identifier::get(getNamespace(), getContext()); */
/*     return OpaqueType::get(ns, parser.getFullSymbolSpec()); */
/*   } */
/*  */
/*   parser.emitError(parser.getNameLoc()) */
/*       << "dialect '" << getNamespace() << "' provides no type parsing hook"; */
/*   return Type(); */
//}

void HILDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  (void)generatedTypePrinter(type, printer);
}
