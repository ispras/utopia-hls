#include "dfcir/DFCIRDialect.h"
#include "dfcir/DFCIRDialect.cpp.inc"

void mlir::dfcir::DFCIRDialect::initialize() {
    registerOperations();
    registerTypes();
}

