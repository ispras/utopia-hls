#ifndef DFCIR_Op_Interfaces
#define DFCIR_Op_Interfaces

include "mlir/IR/OpBase.td"

def NamedOpVal : OpInterface<"NamedOpVal"> {
  let cppNamespace = "mlir::dfcir";

  let methods = [
    InterfaceMethod<"Get the value name",
    "llvm::StringRef", "getValueName", (ins),
    /*methodBody=*/[{ return $_op.getName(); }]>
  ];

}

#endif // DFCIR_Op_Interfaces