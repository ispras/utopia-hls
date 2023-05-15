//===- Dialect.td - HIL dialect ----------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
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
  let summary = "A MLIR dialect for HIL language.";
  let description = [{
      This dialect describes primitives of High-level Intermediate Language
      (HIL) which is an input for the Utopia EDA.
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