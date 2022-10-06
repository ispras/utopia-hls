//===- Dialect.cpp - HIL dialect ---------------*- C++ -*------------------===//
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

#include <iostream>

using namespace mlir;
using namespace hil;

#include "HIL/OpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "HIL/OpsTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#include "HIL/OpsAttributes.cpp.inc"
#undef GET_ATTRDEF_CLASSES

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
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "HIL/OpsAttributes.cpp.inc"
      >();
}

void mlir::hil::PortAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<\"" << getName() << "\" \"" << getTypeName() << "\" <"
      << getFlow() << "> " << getLatency() << ' ' << getIsConst() << ' '
      << getValue() << '>';
}

mlir::Attribute mlir::hil::PortAttr::parse(mlir::AsmParser &parser,
                                                mlir::Type type) {
  if (parser.parseLess())
    return {};
  std::string name;
  if (parser.parseString(&name))
    return {};
  std::string typeName;
  if (parser.parseString(&typeName))
    return {};
  if (parser.parseLess())
    return {};
  double *flow = new double{};
  if (parser.parseFloat(*flow))
    return {};
  if (parser.parseGreater())
    return {};
  unsigned *latency = new unsigned{};
  if (parser.parseInteger(*latency))
    return {};
  unsigned *isConst = new unsigned{};
  if (parser.parseInteger(*isConst))
    return {};
  unsigned int *value = new unsigned{};
  if (parser.parseInteger(*value))
    return {};
  if (parser.parseGreater())
    return {};
  auto *ctx = parser.getContext();
  return get(ctx, name, typeName, flow, *latency, *isConst, *value);
}
