//===- Dialect.td - HIL dialect ----------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HIL_DIALECT
#define HIL_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// HIL dialect definition.
//===----------------------------------------------------------------------===//

def HIL_Dialect : Dialect {
  let name = "hil";
  let summary = "A hil out-of-tree MLIR dialect.";
  let description = [{
      This dialect is an example of an out-of-tree MLIR dialect designed to
      illustrate the basic setup required to develop MLIR-based tools without
      working inside of the LLVM source tree.
  }];
  let cppNamespace = "::mlir::hil";
}

//===----------------------------------------------------------------------===//
// Base HIL operation definition.
//===----------------------------------------------------------------------===//

class HIL_Op<string mnemonic, list<Trait> traits = []> :
        Op<HIL_Dialect, mnemonic, traits>;

class HIL_Type<string name> : TypeDef<HIL_Dialect, name> { }

class HIL_Attr<string name> : AttrDef<HIL_Dialect, name> { }

#endif // HIL_DIALECT
