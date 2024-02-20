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

#include "HIL/Dialect.h"

#include "HIL/Ops.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstring>
#include <iostream>

using AsmPrinter = mlir::AsmPrinter;
using Attribute = mlir::Attribute;
using HILDialect = mlir::hil::HILDialect;
using PortAttr = mlir::hil::PortAttr;
using Type = mlir::Type;

#include "HIL/OpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "HIL/OpsTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#include "HIL/OpsAttributes.cpp.inc"
#undef GET_ATTRDEF_CLASSES

namespace mlir::hil {

llvm::hash_code hash_value(Flow v) {
  uint64_t bytes;
  memcpy(&bytes, &v, sizeof(bytes));
  return llvm::hash_code(bytes);
}

} // namespace mlir::hil

//===----------------------------------------------------------------------===//
// HIL dialect.
//===----------------------------------------------------------------------===//
void HILDialect::initialize() {
registerTypes();
registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "HIL/Ops.cpp.inc"
      >();
}

void HILDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "HIL/OpsTypes.cpp.inc"
      >();
}

void HILDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "HIL/OpsAttributes.cpp.inc"
      >();
}

void PortAttr::print(AsmPrinter &printer) const {
  printer << "<\"" << getName() << "\" \"" << getTypeName() << "\" <"
          << getFlow() << "> " << getLatency() << ' ' << getIsConst() << ' '
          << getValue() << '>';
}

Attribute PortAttr::parse(AsmParser &parser,
                          Type type) {
  std::string name;
  std::string typeName;
  double flow;
  unsigned latency;
  unsigned isConst;
  unsigned value;
  if (parser.parseLess() || parser.parseString(&name) ||
      parser.parseString(&typeName) || parser.parseLess() ||
      parser.parseFloat(flow) || parser.parseGreater() ||
      parser.parseInteger(latency) || parser.parseInteger(isConst) ||
      parser.parseInteger(value) || parser.parseGreater()) {
    llvm::SMLoc loc = parser.getCurrentLocation();
    parser.emitError(loc, "Unable to parse Port! Abort.");
    return Attribute();
  }
  return PortAttr::get(parser.getContext(), name, typeName, Flow(flow), latency, isConst, value);
}
