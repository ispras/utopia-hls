#include "dfcxx/IRbuilders/converter.h"

namespace dfcxx {

DFCIRTypeConverter::DFCIRTypeConverter(mlir::MLIRContext *ctx) : ctx(ctx) {}

mlir::Type DFCIRTypeConverter::operator[](dfcxx::DFVariableImpl *var) {
  const DFTypeImpl &type = var->getType();
  mlir::Type newInnerType;

  if (type.isFixed()) {
    const FixedType &casted = (const FixedType &) (type);
    newInnerType = mlir::dfcir::DFCIRFixedType::get(ctx, casted.isSigned(),
                                                    casted.getIntBits(),
                                                    casted.getFracBits());
  } else if (type.isFloat()) {
    const FloatType &casted = (const FloatType &) (type);
    newInnerType = mlir::dfcir::DFCIRFloatType::get(ctx, casted.getExpBits(),
                                                    casted.getFracBits());
  } else {
    return nullptr;
  }

  if (var->isStream()) {
    return mlir::dfcir::DFCIRStreamType::get(ctx, newInnerType);
  } else if (var->isScalar()) {
    return mlir::dfcir::DFCIRScalarType::get(ctx, newInnerType);
  } else if (var->isConstant()) {
    return mlir::dfcir::DFCIRConstantType::get(ctx, newInnerType);
  }
  return nullptr;
}

} // namespace dfcxx