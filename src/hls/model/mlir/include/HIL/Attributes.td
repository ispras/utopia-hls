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

def InputArgAttr : HIL_Attr<"InputArg"> {
  let mnemonic = "input";

  let summary = "Input argument";

  let description = [{
    Input arg type
  }];

  let parameters = (ins
    "std::string":$typeName,
    "double*":$flow,
    "std::string":$name);

  let hasCustomAssemblyFormat = 1;
}

def OutputArgAttr : HIL_Attr<"OutputArg"> {
  let mnemonic = "output";

  let summary = "Output argument";

  let description = [{
    Output arg type
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
