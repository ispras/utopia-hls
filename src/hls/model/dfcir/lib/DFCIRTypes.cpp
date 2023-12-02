#include "DFCIRTypes.h"

#include "DFCIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace dfcir;

#define GET_TYPEDEF_CLASSES
#include "DFCIRTypes.cpp.inc"

void DFCIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "DFCIRTypes.cpp.inc"
      >();
}
