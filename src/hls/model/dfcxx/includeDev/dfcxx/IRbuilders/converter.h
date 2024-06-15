#ifndef DFCXX_IR_BUILDER_CONVERTER_H
#define DFCXX_IR_BUILDER_CONVERTER_H

#include "dfcir/DFCIROperations.h"
#include "dfcxx/kernel.h"
#include "dfcxx/typedefs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace dfcxx {

class DFCIRTypeConverter {
  mlir::MLIRContext *ctx;

public:
  DFCIRTypeConverter(mlir::MLIRContext *ctx);

  mlir::Type operator[](DFVariableImpl *var);
};

} // namespace dfcxx

#endif // DFCXX_IR_BUILDER_CONVERTER_H
