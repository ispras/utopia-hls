//===- Attributes.td - HIL dialect attributes --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HIL_ATTRIBUTES
#define HIL_ATTRIBUTES

include "Dialect.td"

def InputPortAttr : HIL_Attr<"InputPort"> {
  let mnemonic = "input";

  let summary = "Input port argument";

  let description = [{
    Input port arg type
  }];

  let parameters = (ins
    "std::string":$typeName,
    "double*":$flow,
    "std::string":$name);

  let hasCustomAssemblyFormat = 1;
}

def OutputPortAttr : HIL_Attr<"OutputPort"> {
  let mnemonic = "output";

  let summary = "Output port argument";

  let description = [{
    Output port arg type
  }];

  let parameters = (ins
    "std::string":$typeName,
    "double*":$flow,
    "unsigned":$latency,
    "std::string":$name,
    "std::string":$value);

  let hasCustomAssemblyFormat = 1;
}

#endif // HIL_ATTRIBUTES
