#include "DFCIRDialect.h"
#include "DFCIROperations.h"
#include "DFCIRTypes.h"

using namespace mlir;
using namespace dfcir;

#include "DFCIRDialect.cpp.inc"

void DFCIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DFCIROperations.cpp.inc"
      >();
  registerTypes();
}
