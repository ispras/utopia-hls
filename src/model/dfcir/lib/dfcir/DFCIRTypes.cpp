#include "dfcir/DFCIRDialect.h"
#include "dfcir/DFCIRTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "dfcir/DFCIRTypeInterfaces.cpp.inc" // Cannot enforce header sorting.

void mlir::dfcir::DFCIRFixedType::printSVSignature(llvm::raw_string_ostream &out) {
  out << "fix_" << getSign() << "_" << getIntegerBits() << "_" << getFractionBits();
}

uint64_t mlir::dfcir::DFCIRFixedType::getBitWidth() {
  return getIntegerBits() + getFractionBits();
}

void mlir::dfcir::DFCIRFloatType::printSVSignature(llvm::raw_string_ostream &out) {
  out << "flt_" << getExponentBits() << "_" << getFractionBits();
}

uint64_t mlir::dfcir::DFCIRFloatType::getBitWidth() {
  return getExponentBits() + getFractionBits();
}

void mlir::dfcir::DFCIRRawBitsType::printSVSignature(llvm::raw_string_ostream &out) {
  out << "bits_" << getBits();
}

uint64_t mlir::dfcir::DFCIRRawBitsType::getBitWidth() {
  return getBits();
}

mlir::Type mlir::dfcir::DFCIRStreamType::getDFType() {
  return getStreamType();
}

mlir::Type mlir::dfcir::DFCIRScalarType::getDFType() {
  return getScalarType();
}

mlir::Type mlir::dfcir::DFCIRConstantType::getDFType() {
  return getConstType();
}

#define GET_TYPEDEF_CLASSES

#include "dfcir/DFCIRTypes.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST

#include "dfcir/DFCIRTypes.cpp.inc"

  >();
}
