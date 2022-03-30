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

/* Type HILDialect::parseType(DialectAsmParser &parser) const { */
/*   StringRef data_type; */
/*   if (parser.parseKeyword(&data_type)) */
/*     return {}; */
/*   Type value; */
/*   generatedTypeParser(parser, data_type, value); */
/*   return value; */
/* } */

/* void HILDialect::printType(Type type, */
/*                            DialectAsmPrinter &printer) const { */
/*   (void)generatedTypePrinter(type, printer); */
/* } */

/*
Attribute HILDialect::parseAttribute(DialectAsmParser &parser,
                         Type type) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type))
    return {};
  Attribute value;
  generatedAttributeParser(parser, data_type, type, value);
  return value;
}

void HILDialect::printAttribute(Attribute attr,
                           DialectAsmPrinter &printer) const {
  (void)generatedAttributePrinter(attr, printer);
}
*/

void mlir::hil::InputArgAttr::print(mlir::AsmPrinter &printer) const {
  printer << "input<\"" <<
      getTypeName() <<
      "\"<" << getFlow()
      << ">" << " " << '"' << getName() << '"' << ">";
}

mlir::Attribute mlir::hil::InputArgAttr::parse(mlir::AsmParser &parser,
                                               mlir::Type type) {
  if (parser.parseLess())
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
    std::string name;
    if (parser.parseString(&name))
      return {};
    if (parser.parseGreater())
      return {};
    return get(parser.getContext(), typeName, flow, name);
}

void mlir::hil::OutputArgAttr::print(mlir::AsmPrinter &printer) const {
  printer << "output<\"" << getTypeName() << "\"<" << getFlow()
      << ">" << " " << getLatency() << " " << '"' << getName()
      << '"' << (getValue().empty() ? "" : " = \"" + getValue() + "\"") << ">";
}

mlir::Attribute mlir::hil::OutputArgAttr::parse(mlir::AsmParser &parser,
                                                mlir::Type type) {
  if (parser.parseLess())
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
    unsigned latency;
    if (parser.parseInteger(latency))
      return {};
    std::string name;
    if (parser.parseString(&name))
      return {};
    if (parser.parseOptionalEqual()) {
      if (parser.parseGreater())
        return {};
      return get(parser.getContext(), typeName, flow, latency, name, "");
    }
    std::string value;
    if (parser.parseString(&value))
      return {};
    if (parser.parseGreater())
      return {};
    return get(parser.getContext(), typeName, flow, latency, name, value);
}

