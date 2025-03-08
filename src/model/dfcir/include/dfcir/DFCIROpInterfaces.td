//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_Op_Interfaces
#define DFCIR_Op_Interfaces

include "mlir/IR/OpBase.td"

def NamedOpVal : OpInterface<"NamedOpVal"> {
  let cppNamespace = "mlir::dfcir";

  let methods = [
    InterfaceMethod<
      "Get the value name",
      "llvm::StringRef",
      "getValueName",
      (ins),
      /*methodBody=*/[{ return $_op.getName(); }],
      /*defaultImplementation=*/[{}]
    >
  ];
}

def InputOutputOpInterface : OpInterface<"InputOutputOpInterface"> {
  let cppNamespace = "mlir::dfcir";
}

def InputOpInterface : OpInterface<"InputOpInterface", [InputOutputOpInterface]> {
  let cppNamespace = "mlir::dfcir";
}

def OutputOpInterface : OpInterface<"OutputOpInterface", [InputOutputOpInterface]> {
  let cppNamespace = "mlir::dfcir";
}

def NaryOpInterface : OpInterface<"NaryOpInterface"> {
  let cppNamespace = "mlir::dfcir";
}

def UnaryOpInterface : OpInterface<"UnaryOpInterface", [NaryOpInterface]> {
  let cppNamespace = "mlir::dfcir";
}

def BinaryOpInterface : OpInterface<"BinaryOpInterface", [NaryOpInterface]> {
  let cppNamespace = "mlir::dfcir";
}

def ShiftOpInterface : OpInterface<"ShiftOpInterface"> {
  let cppNamespace = "mlir::dfcir";
}

def ConstantInputInterface : OpInterface<"ConstantInputInterface"> {
  let cppNamespace = "mlir::dfcir";
}

def Scheduled : OpInterface<"Scheduled"> {
  let cppNamespace = "mlir::dfcir";

  let methods = [
    InterfaceMethod<
      "Get clock latency (or -1 if not set).",
      "int32_t",
      "getLatency",
      (ins),
      /*methodBody=*/[{
        return static_cast<int32_t>(getLatencyAttr(impl, $_op).getSInt());
      }],
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<
      "Get clock latency as an attribute.",
      "mlir::IntegerAttr",
      "getLatencyAttr",
      (ins),
      /*methodBody=*/[{ return $_op.getLatencyAttr(); }],
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<
      "Set clock latency (-1 means it is not set).",
      "void",
      "setLatency",
      (ins "int32_t":$latency),
      /*methodBody=*/[{
        auto attr = 
            mlir::IntegerAttr::get(IntegerType::get($_op.getContext(), 32),
                                   latency);
        return setLatencyAttr(impl, $_op, attr);
      }],
      /*defaultImplementation=*/[{}]
    >,
    InterfaceMethod<
      "Set clock latency as an attribute.",
      "void",
      "setLatencyAttr",
      (ins "mlir::IntegerAttr":$latency),
      /*methodBody=*/[{ return $_op.setLatencyAttr(latency); }],
      /*defaultImplementation=*/[{}]
    >
  ];
}

#endif // DFCIR_Op_Interfaces
