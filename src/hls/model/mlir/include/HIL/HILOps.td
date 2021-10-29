//===- HILOps.td - HIL dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "HILDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HIL_Model : HIL_Op<"model", [NoTerminator]> {
  let summary = "Model container.";
  let description = [{
    The `hil.model` operation
  }];

  let regions = (region AnyRegion:$region);
  let assemblyFormat = [{
    $region attr-dict
  }];
}

def InputArgType : HIL_Type<"HILInputArg"> {
  let mnemonic = "input";

  let summary = "Input argument";

  let description = [{
    Input arg type
  }];

  let parameters = (ins
    "std::string":$typeName,
    "int64_t":$flow,
    "std::string":$name);

  let printer = [{
    int64_t uncasted_flow = getImpl()->flow;
    $_printer << "input<\"" <<
      getImpl()->typeName << "\"<" << *reinterpret_cast<double*>(&uncasted_flow) << ">" << " " << '"' << getImpl()->name << '"' << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    std::string typeName;
    if ($_parser.parseString(&typeName))
      return Type();
    if ($_parser.parseLess())
      return Type();
    double inner_flow;
    if ($_parser.parseFloat(inner_flow))
      return Type();
    int64_t flow = *reinterpret_cast<int64_t*>(&inner_flow);
    if ($_parser.parseGreater())
      return Type();
    std::string name;
    if ($_parser.parseString(&name))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, typeName, flow, name);
  }];
}

def OutputArgType : HIL_Type<"HILOutputArg"> {
  let mnemonic = "output";

  let summary = "Output argument";

  let description = [{
    Output arg type
  }];

  let parameters = (ins
    "std::string":$typeName,
    "int64_t":$flow,
    "int":$bind,
    "std::string":$name);

  let printer = [{
    int64_t uncasted_flow = getImpl()->flow;
    $_printer << "output<\"" << getImpl()->typeName << "\"<" <<  *reinterpret_cast<double*>(&uncasted_flow) << ">" << " " << getImpl()->bind << " " << '"' << getImpl()->name << '"' << ">";
  }];

  let parser = [{
    if ($_parser.parseLess())
      return Type();
    std::string typeName;
    if ($_parser.parseString(&typeName))
      return Type();
    if ($_parser.parseLess())
      return Type();
    double inner_flow;
    if ($_parser.parseFloat(inner_flow))
      return Type();
    int64_t flow = *reinterpret_cast<int64_t*>(&inner_flow);
    if ($_parser.parseGreater())
      return Type();
    int bind;
    if ($_parser.parseInteger(bind))
      return Type();
    std::string name;
    if ($_parser.parseString(&name))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, typeName, flow, bind, name);
  }];
}

def InputArgTypeAttr : TypeAttrBase<"HILInputArgType", "input arg attribute">;
def OutputArgTypeAttr : TypeAttrBase<"HILOutputArgType", "output arg attribute">;

def HIL_NodeType : HIL_Op<"nodetype", [NoSideEffect]> {
  let summary = "Node type.";
  let description = [{
    The `hil.nodetype` operation for defining types of nodes
  }];

  let arguments = (ins
    StrAttr:$name,
    TypedArrayAttrBase<InputArgTypeAttr, "input args">:$commandArguments,
    TypedArrayAttrBase<OutputArgTypeAttr, "output args">:$commandResults);
  let assemblyFormat = [{
    $name $commandArguments `=``>` $commandResults attr-dict
  }];
}

def HIL_Node : HIL_Op<"node", [NoSideEffect]> {
  let summary = "Node.";
  let description = [{
    The `hil.nodetype` operation for defining types of nodes
  }];

  let arguments = (ins
    StrAttr:$nodeTypeName,
    StrAttr:$name,
    StrArrayAttr:$commandArguments,
    StrArrayAttr:$commandResults);
  let assemblyFormat = [{
    $nodeTypeName $name $commandArguments `=``>` $commandResults attr-dict
  }];
}

def HIL_Chan : HIL_Op<"chan", [NoSideEffect]> {
  let summary = "Channel.";
  let description = [{
    The `hil.nodetype` operation for defining types of nodes
  }];

  let arguments = (ins
    StrAttr:$typeName,
    StrAttr:$varName);
  let assemblyFormat = [{
    $typeName $varName attr-dict
  }];
}

#endif // STANDALONE_OPS
