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

def PortAttr : HIL_Attr<"Port"> {
  let mnemonic = "port";
  let summary = "Port attribute";

  let description = [{
    port arg type
  }];

  let parameters = (ins
    "std::string":$name,
    "std::string":$typeName,
    "double*":$flow,
    "unsigned*":$latency,
    "bool*":$isConst,
    "unsigned*":$value);

  let hasCustomAssemblyFormat = 1;
}

def BindingAttr : HIL_Attr<"Binding"> {

  let mnemonic = "binding";
  let summary = "Binding attribute";

  let description = [{
    binding attr type
  }];

  let parameters = (ins
    "std::string":$nodeName,
    PortAttr:$port);

  let hasCustomAssemblyFormat = 1;
}

#endif // HIL_ATTRIBUTES
