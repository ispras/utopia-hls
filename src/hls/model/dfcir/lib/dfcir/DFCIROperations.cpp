#include "dfcir/DFCIROperations.h"
#include "dfcir/DFCIRDialect.h"

#define GET_OP_CLASSES
#include "dfcir/DFCIROperations.cpp.inc"

void mlir::dfcir::DFCIRDialect::registerOperations() {
    addOperations<
#define GET_OP_LIST
#include "dfcir/DFCIROperations.cpp.inc"
    >();
}
