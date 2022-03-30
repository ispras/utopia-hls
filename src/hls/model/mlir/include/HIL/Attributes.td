//===- Attributes.td - HIL dialect attributes --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "Ops.td"

#ifndef HIL_ATTRIBUTES
#define HIL_ATTRIBUTES

/*def InputArgAttr : HIL_Attr<"InputArg"> {
  let mnemonic = "input";

  let summary = "Input argument";

  let description = [{
    Input arg type
  }];

  let parameters = (ins
    "std::string":$typeName,
    "double*":$flow,
    "std::string":$name);

  let printer = [{
    $_printer << "input<\"" <<
      getImpl()->typeName << "\"<" << *getImpl()->flow << ">" << " " << '"' << getImpl()->name << '"' << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return {};
    std::string typeName;
    if ($_parser.parseString(&typeName))
      return {};
    if ($_parser.parseLess())
      return {};
    double *flow = new double{};
    if ($_parser.parseFloat(*flow))
      return {};
    if ($_parser.parseGreater())
      return {};
    std::string name;
    if ($_parser.parseString(&name))
      return {};
    if ($_parser.parseGreater())
      return {};
    return get($_ctxt, typeName, flow, name);
  }];
}*/

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
