#ifndef DFCIR_Dialect
#define DFCIR_Dialect

include "mlir/IR/DialectBase.td"

def DFCIR_Dialect : Dialect {

  // CPP namespace is implied to be the same by default.
  let name = "dfcir";
  
  let cppNamespace = "::mlir::dfcir";
  
  let summary = "A high-level dataflow core description representation.";

  let extraClassDeclaration = [{
    void registerOperations();
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // DFCIR_Dialect
