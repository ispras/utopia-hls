//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/DFCIRDialect.h"
#include "dfcir/DFCIRTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "dfcir/DFCIRTypeInterfaces.cpp.inc" // Cannot enforce header sorting.

namespace mlir::dfcir {

void DFCIRFixedType::printSVSignature(llvm::raw_string_ostream &out) {
  out << "fix_" << getSign() << "_" 
      << getIntegerBits() << "_" << getFractionBits();
}

uint64_t DFCIRFixedType::getBitWidth() {
  return getIntegerBits() + getFractionBits();
}

void DFCIRFloatType::printSVSignature(llvm::raw_string_ostream &out) {
  out << "flt_" << getExponentBits() << "_" << getFractionBits();
}

uint64_t DFCIRFloatType::getBitWidth() {
  return getExponentBits() + getFractionBits();
}

void DFCIRRawBitsType::printSVSignature(llvm::raw_string_ostream &out) {
  out << "bits_" << getBits();
}

uint64_t DFCIRRawBitsType::getBitWidth() {
  return getBits();
}

Type DFCIRStreamType::getDFType() {
  return getStreamType();
}

Type DFCIRScalarType::getDFType() {
  return getScalarType();
}

Type DFCIRConstantType::getDFType() {
  return getConstType();
}

} // namespace mlir::dfcir

#define GET_TYPEDEF_CLASSES

#include "dfcir/DFCIRTypes.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST

#include "dfcir/DFCIRTypes.cpp.inc"

  >();
}
