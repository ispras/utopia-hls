//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::standalone;

#include "Standalone/StandaloneOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "Standalone/StandaloneOpsTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//
void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Standalone/StandaloneOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Standalone/StandaloneOpsTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
    >();
}

Type StandaloneDialect::parseType(DialectAsmParser &parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();
  mlir::Type value;
  generatedTypeParser(parser, data_type, value);
  return value;
}
  //return StandaloneLolType::parse(parser);
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

void StandaloneDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  generatedTypePrinter(type, printer);
}
