//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/DFCIRDialect.h"
#include "dfcir/DFCIROperations.h"

namespace mlir::dfcir {

ParseResult ScalarInputOp::parse(OpAsmParser &parser,
                                 OperationState &result) {
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);
  StringAttr nameAttr;
  
  if (parser.parseLess()) { return failure(); }

  Type scalarType;
  if (parser.parseCustomTypeWithFallback(scalarType)) { return failure(); }
  DFCIRScalarType type = DFCIRScalarType::get(result.getContext(), scalarType);
  resRawTypes[0] = type;

  if (parser.parseGreater() || parser.parseLParen()) { return failure(); }

  if (parser.parseCustomAttributeWithFallback(
      nameAttr,
      parser.getBuilder().getType<NoneType>())) { return failure(); }
  if (nameAttr) result.attributes.append("name", nameAttr);

  if (parser.parseRParen()) { return failure(); }

  if (parser.parseOptionalAttrDict(result.attributes)) { return failure(); }
  
  result.addTypes(resTypes);
  return success();
}

void ScalarInputOp::print(OpAsmPrinter &printer) {
  printer << "<";
  printer << getRes().getType().getScalarType();
  printer << ">" << ' ' << "(";
  printer.printAttributeWithoutType(getNameAttr());
  printer << ")";
  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

llvm::StringRef ScalarInputOp::getValueName() {
  return getName();
}

ParseResult ScalarOutputOp::parse(OpAsmParser &parser,
                                  OperationState &result) {
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);
  StringAttr nameAttr;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> streamOperands;
  llvm::SMLoc streamOperandsLoc;
  llvm::SmallVector<Type, 1> streamTypes;
  
  if (parser.parseLess()) { return failure(); }

  Type scalarType;
  if (parser.parseCustomTypeWithFallback(scalarType)) { return failure(); }
  DFCIRScalarType type = DFCIRScalarType::get(result.getContext(), scalarType);
  resRawTypes[0] = type;

  if (parser.parseGreater() || parser.parseLParen()) { return failure(); }

  if (parser.parseCustomAttributeWithFallback(
      nameAttr,
      parser.getBuilder().getType<NoneType>())) { return failure(); }
  if (nameAttr) result.attributes.append("name", nameAttr);
  
  if (parser.parseRParen()) { return failure(); }

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseEqual()) { return failure(); }

    streamOperandsLoc = parser.getCurrentLocation();
    OpAsmParser::UnresolvedOperand operand;
    OptionalParseResult parseResult = parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
      streamOperands.push_back(operand);
    }
    
    if (parser.parseColon()) { return failure(); }

    Type optionalType;
    parseResult = parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
      streamTypes.push_back(optionalType);
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) { return failure(); }

  result.addTypes(resTypes);
  if (parser.resolveOperands(streamOperands, streamTypes, streamOperandsLoc,
                             result.operands)) { return failure(); }
  return success();
}

void ScalarOutputOp::print(OpAsmPrinter &printer) {
  printer << "<";
  printer << getRes().getType().getScalarType();
  printer << ">" << ' ' << "(";
  printer.printAttributeWithoutType(getNameAttr());
  printer << ")";
  if (Value value = getStream()) {
    printer << ' ' << "<" << "=" << ' ';
    printer << value;
    printer << ' ' << ":" << ' ';
    printer << llvm::ArrayRef<Type>(value.getType());
  }
  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

llvm::StringRef ScalarOutputOp::getValueName() {
  return getName();
}

ParseResult InputOp::parse(OpAsmParser &parser, OperationState &result) {
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);
  StringAttr nameAttr;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> ctrlOperands;
  llvm::SMLoc ctrlOperandsLoc;
  llvm::SmallVector<Type, 1> ctrlTypes;

  if (parser.parseLess()) { return failure(); }
  
  Type streamType;
  if (parser.parseCustomTypeWithFallback(streamType)) { return failure(); }
  DFCIRStreamType type = DFCIRStreamType::get(result.getContext(), streamType);
  resRawTypes[0] = type;

  if (parser.parseGreater() || parser.parseLParen()) { return failure(); }

  if (parser.parseCustomAttributeWithFallback(
      nameAttr,
      parser.getBuilder().getType<NoneType>())) { return failure(); }
  if (nameAttr) result.attributes.append("name", nameAttr);
  
  if (succeeded(parser.parseOptionalComma())) {
    ctrlOperandsLoc = parser.getCurrentLocation();
    OpAsmParser::UnresolvedOperand operand;
    OptionalParseResult parseResult = parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
      ctrlOperands.push_back(operand);
    }
    
    if (parser.parseColon()) { return failure(); }

    Type optionalType;
    parseResult = parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
      ctrlTypes.push_back(optionalType);
    }
  }

  if (parser.parseRParen()) { return failure(); }
  
  if (parser.parseOptionalAttrDict(result.attributes)) { return failure(); }

  result.addTypes(resTypes);
  if (parser.resolveOperands(ctrlOperands, ctrlTypes, ctrlOperandsLoc,
                             result.operands)) { return failure(); }
  return success();
}

void InputOp::print(OpAsmPrinter &printer) {
  printer << "<";
  printer << getRes().getType().getStreamType();
  printer << ">" << ' ' << "(";
  printer.printAttributeWithoutType(getNameAttr());
  if (Value value = getCtrl()) {
    printer << "," << ' ';
    printer << value;
    printer << ' ' << ":" << ' ';
    printer << llvm::ArrayRef<Type>(value.getType());
  }
  printer << ")";
  llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

StringRef InputOp::getValueName() {
  return getName();
}

ParseResult OutputOp::parse(OpAsmParser &parser, OperationState &result) {
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);
  StringAttr nameAttr;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> ctrlOperands;
  llvm::SMLoc ctrlOperandsLoc;
  llvm::SmallVector<Type, 1> ctrlTypes;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> streamOperands;
  llvm::SMLoc streamOperandsLoc;
  llvm::SmallVector<Type, 1> streamTypes;
  
  if (parser.parseLess()) { return failure(); }

  Type streamType;
  if (parser.parseCustomTypeWithFallback(streamType)) { return failure(); }
  DFCIRStreamType type = DFCIRStreamType::get(result.getContext(), streamType);
  resRawTypes[0] = type;

  if (parser.parseGreater() || parser.parseLParen()) { return failure(); }

  if (parser.parseCustomAttributeWithFallback(
      nameAttr,
      parser.getBuilder().getType<NoneType>())) { return failure(); }
  if (nameAttr) result.attributes.append("name", nameAttr);

  if (succeeded(parser.parseOptionalComma())) {
    ctrlOperandsLoc = parser.getCurrentLocation();
    OpAsmParser::UnresolvedOperand operand;
    OptionalParseResult parseResult = parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
      ctrlOperands.push_back(operand);
    }
    
    if (parser.parseColon()) { return failure(); }

    Type optionalType;
    parseResult = parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
        ctrlTypes.push_back(optionalType);
    }
  }

  if (parser.parseRParen()) { return failure(); }

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseEqual()) { return failure(); }

    streamOperandsLoc = parser.getCurrentLocation();
    OpAsmParser::UnresolvedOperand operand;
    OptionalParseResult parseResult = parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
      streamOperands.push_back(operand);
    }

    if (parser.parseColon()) { return failure(); }

    Type optionalType;
    parseResult = parser.parseOptionalType(optionalType);
    if (parseResult.has_value()) {
      if (failed(*parseResult)) { return failure(); }
      streamTypes.push_back(optionalType);
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) { return failure(); }

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                              {static_cast<int32_t>(ctrlOperands.size()),
                               static_cast<int32_t>(streamOperands.size())}));
  result.addTypes(resTypes);
  if (parser.resolveOperands(ctrlOperands, ctrlTypes, ctrlOperandsLoc,
                             result.operands)) { return failure(); }
  if (parser.resolveOperands(streamOperands, streamTypes, streamOperandsLoc,
                             result.operands)) { return failure(); }
  return success();
}

void OutputOp::print(OpAsmPrinter &printer) {
  printer << "<";
  printer << getRes().getType().getStreamType();
  printer << ">" << ' ' << "(";
  printer.printAttributeWithoutType(getNameAttr());
  if (Value value = getCtrl()) {
    printer << "," << ' ';
    printer << value;
    printer << ' ' << ":" << ' ';
    printer << llvm::ArrayRef<Type>(value.getType());
  }
  printer << ")";
  if (Value value = getStream()) {
    printer << ' ' << "<" << "=" << ' ';
    printer << value;
    printer << ' ' << ":" << ' ';
    printer << llvm::ArrayRef<Type>(value.getType());
  }
  llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("name");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

llvm::StringRef OutputOp::getValueName() {
  return getName();
}

ParseResult MuxOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand control;
  SmallVector<OpAsmParser::UnresolvedOperand, 16> vars;
  SmallVector<Type, 16> types;
  Type controlType, resType;

  if (parser.parseLParen() || parser.parseOperand(control) ||
      parser.parseColon() || parser.parseType(controlType) || 
      parser.parseComma()) { return failure(); }

  llvm::SMLoc varLocation = parser.getCurrentLocation();
  do {
    OpAsmParser::UnresolvedOperand var;
    Type varType;
    if (parser.parseOperand(var)) {
      return failure();
    }
    vars.push_back(var);

    if (succeeded(parser.parseOptionalColon()) && parser.parseType(varType)) {
      return failure();
    }

    types.push_back(varType);
  } while(succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen() || parser.parseColon() ||
      parser.parseType(resType) ||
      parser.parseOptionalAttrDict(result.attributes)) { return failure(); }

  if (parser.resolveOperand(control, controlType, 
                            result.operands)) { return failure(); }

  result.addTypes(resType);
  for (Type &type: types) {
    if (!type) {
      type = resType;
    }
  }
  return parser.resolveOperands(vars, types, varLocation, result.operands);
}

void MuxOp::print(OpAsmPrinter &p) {
  Value value = getControl();
  p << "(" << value << ": " << value.getType();

  auto vars = getVars();
  auto resType = getType();
  for (const auto &var: vars) {
    p << ", " << var;
    auto type = var.getType();
    if (type != resType) {
      p << ": " << type;
    }
  }
  p << ") : " << resType;
  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);
  Attribute valueAttr;

  if (parser.parseLess()) { return failure(); }

  Type constType;
  if (parser.parseCustomTypeWithFallback(constType)) { return failure(); }
  DFCIRConstantType type = DFCIRConstantType::get(result.getContext(), 
                                                  constType);
  resRawTypes[0] = type;

  if (parser.parseGreater()) { return failure(); }

  if (parser.parseAttribute(valueAttr, Type{})) { return failure(); }
  if (valueAttr) result.attributes.append("value", valueAttr);

  if (parser.parseOptionalAttrDict(result.attributes)) { return failure(); }
  result.addTypes(resTypes);
  return success();
}

void ConstantOp::print(OpAsmPrinter &printer) {
  printer << "<";
  printer << getRes().getType().getConstType();
  printer << ">" << ' ';
  printer.printAttribute(getValueAttr());
  llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

ParseResult OffsetOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand strRawOpers[1];
  llvm::ArrayRef<OpAsmParser::UnresolvedOperand> strOpers(strRawOpers);
  llvm::SMLoc streamOperandsLoc;
  Type streamRawTypes[1];
  llvm::ArrayRef<Type> streamTypes(streamRawTypes);
  IntegerAttr offsetAttr;
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);

  if (parser.parseLParen()) { return failure(); }

  streamOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strRawOpers[0]) || parser.parseComma() ||
      parser.parseCustomAttributeWithFallback(offsetAttr, 
                                              Type{})) { return failure(); }

  if (offsetAttr)
    result.getOrAddProperties<OffsetOp::Properties>().offset = offsetAttr;
  
  if (parser.parseRParen()) { return failure(); }
  
  auto loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes)) { return failure(); }
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
    return parser.emitError(loc) << "'" 
                                 << result.name.getStringRef() << "' op ";
    }))) { return failure(); }

  if (parser.parseColon()) { return failure(); }

  DFCIRStreamType type;
  if (parser.parseCustomTypeWithFallback(type)) { return failure(); }
  resRawTypes[0] = type;
  streamRawTypes[0] = type;

  result.addTypes(resTypes);
  if (parser.resolveOperands(strOpers, streamTypes, streamOperandsLoc,
                             result.operands)) { return failure(); }
  return success();
}

void mlir::dfcir::OffsetOp::print(OpAsmPrinter &printer) {
  printer << "(" << getStream() << ", ";
  printer.printStrippedAttrOrType(getOffsetAttr());
  printer << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("offset");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << " : " << getRes().getType();
}

ParseResult LatencyOp::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr latencyAttr;
  OpAsmParser::UnresolvedOperand inputRawOperands[1];
  llvm::ArrayRef<OpAsmParser::UnresolvedOperand> inputOperands(inputRawOperands);
  llvm::SMLoc inputOperandsLoc;
  Type inputRawTypes[1];
  llvm::ArrayRef<Type> inputTypes(inputRawTypes);
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);
  if (parser.parseLSquare() ||
      parser.parseCustomAttributeWithFallback(latencyAttr, Type{})) {
    return failure();
  }
  
  if (latencyAttr) {
    result.getOrAddProperties<LessOp::Properties>().latency = latencyAttr;
  }

  if (parser.parseRSquare() || 
      parser.parseLParen()) {
    return failure();
  }

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]) ||
      parser.parseColon() ||
      parser.parseCustomTypeWithFallback(inputRawTypes[0]) ||
      parser.parseRParen() ||
      parser.parseColon() ||
      parser.parseCustomTypeWithFallback(resRawTypes[0]) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands)) {
    return failure();
  }
  return success();
}

void LatencyOp::print(OpAsmPrinter &printer) {
  printer << "[" << getLatency() << "] (";
  printer << getInput() << " : " << getInput().getType() << ") : ";
  printer << getRes().getType();
  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

ParseResult SimpleCounterOp::parse(OpAsmParser &parser, OperationState &result) {
  Type resRawTypes[1];
  llvm::ArrayRef<Type> resTypes(resRawTypes);
  OpAsmParser::UnresolvedOperand maxRawOperands[1];
  llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> maxOperands(maxRawOperands);
  llvm::SMLoc maxOperandsLoc;

  Type maxRawTypes[1];
  llvm::ArrayRef<::mlir::Type> maxTypes(maxRawTypes);
  if (parser.parseLess())
    return failure();

  Type streamType;
  if (parser.parseCustomTypeWithFallback(streamType)) { return failure(); }
  DFCIRStreamType type = DFCIRStreamType::get(result.getContext(), streamType);
  resRawTypes[0] = type;
  if (parser.parseGreater() || parser.parseLParen())
    return failure();
  maxOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maxRawOperands[0]))
    return failure();
  if (parser.parseColon())
    return failure();
  DFCIRScalarType maxType;
  if (parser.parseCustomTypeWithFallback(maxType)) {
    return failure();
  }
  maxRawTypes[0] = maxType;
  if (parser.parseRParen())
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(resTypes);
  if (parser.resolveOperands(maxOperands, maxTypes, maxOperandsLoc, result.operands))
    return ::mlir::failure();
  return success();
}

void SimpleCounterOp::print(::mlir::OpAsmPrinter &printer) {
  printer << "<";
  printer << getRes().getType().getStreamType();
  printer << "> (";
  printer << getMax() << ": " << getMax().getType();
  printer << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("max");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}


} // namespace mlir::dfcir

#define GET_OP_CLASSES

#include "dfcir/DFCIROperations.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST

#include "dfcir/DFCIROperations.cpp.inc"

  >();
}
