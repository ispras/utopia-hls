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
  let mnemonic = "input-port";

  let summary = "Input port";

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
  let mnemonic = "output-port";

  let summary = "Output port";

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

def InputBndAttr : HIL_Attr<"InputBnd"> {

  let mnemonic = "input-binding";

  let summary = "Input port binding";

  let description = [{
    Input binding attr type
  }];

  let parameters = (ins
    "std::string":$nodeName,
    InputPortAttr:$port);

  let hasCustomAssemblyFormat = 1;
}

def OutputBndAttr : HIL_Attr<"OutputBnd"> {

  let mnemonic = "output-binding";

  let summary = "Output port binding";

  let description = [{
    Output binding attr type
  }];

  let parameters = (ins
    "std::string":$nodeName,
    OutputPortAttr:$port);

  let hasCustomAssemblyFormat = 1;
}

#endif // HIL_ATTRIBUTES
