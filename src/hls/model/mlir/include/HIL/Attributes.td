//===- Attributes.td - HIL dialect attributes --------------*- tablegen -*-===//
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
    "Flow":$flow,
    "unsigned":$latency,
    "unsigned":$isConst,
    "unsigned":$value);

  let hasCustomAssemblyFormat = 1;
}

def BindingAttr : HIL_Attr<"Binding"> {
  let mnemonic = "bnd";
  let summary = "Binding attribute";

  let description = [{
    binding attr type
  }];

  let parameters = (ins
    StringRefParameter<>:$nodeName,
    PortAttr:$port);

  let assemblyFormat = "`<` $nodeName qualified($port) `>`";
}

def BindingGraphAttr : HIL_Attr<"BindingGraph"> {
  let mnemonic = "bndgraph";
  let summary = "BindingGraph attribute";

  let description = [{
    binding graph attr type
  }];

  let parameters = (ins
    StringRefParameter<>:$graphName,
    StringRefParameter<>:$chanName);

  let assemblyFormat = "`<` $graphName $chanName `>`";
}

#endif // HIL_ATTRIBUTES