//===- HILOps.td - HIL dialect operations ------------------*- tablegen -*-===//
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

#ifndef HIL_OPS
#define HIL_OPS

include "Attributes.td"

class HIL_Container<string mnemonic, list<Trait> traits = []> :
  HIL_Op<mnemonic, !listconcat(traits, [
    NoRegionArguments,
    NoTerminator,
    SingleBlock
  ])> {
  let assemblyFormat = "$body attr-dict";
  let regions = (region SizedRegion<1>: $body);
  let extraClassDeclaration = [{
  Block *getBody() {
    Region& region = getOperation()->getRegion(0);
    assert(region.hasOneBlock() && "The body should have one block.");
    return &region.front();
  }
}];
}

def HIL_Model : HIL_Container<"model"> {
  let summary = "Model container.";
  let description = [{
    The `hil.model` operation
  }];
  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name $body attr-dict";
}

def HIL_Graphs : HIL_Container<"graphs", [HasParent<"Model">]> {
  let summary = "Graphs container.";
  let description = [{
    Container of `hil.graph`
  }];
}

def HIL_Insts : HIL_Container<"insts", [HasParent<"Graph">]> {
  let summary = "Insts container.";
  let description = [{
    Container of `hil.inst`
  }];
}

def HIL_Graph : HIL_Container<"graph", [HasParent<"Graphs">]> {
  let summary = "Graph.";
  let description = [{
    The `hil.graph` operation
  }];

  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name $body attr-dict";
}

def HIL_NodeTypes : HIL_Container<"nodetypes", [HasParent<"Model">]> {
  let summary = "Nodetypes container.";
  let description = [{
    Container of `hil.nodetype`
  }];
}

def HIL_Inst : HIL_Container<"inst", [HasParent<"Insts">]> {
  let summary = "Instance.";
  let description = [{
    The `hil.inst` operation
  }];

  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name $body attr-dict";
}

def HIL_Chans : HIL_Container<"chans", [HasParent<"Graph">]> {
  let summary = "Chans container.";
  let description = [{
    Container of `hil.chan`
  }];
}

def HIL_Cons : HIL_Container<"cons", [HasParent<"Inst">]> {
  let summary = "Cons container.";
  let description = [{
    Container of `hil.con`
  }];
}

def HIL_Nodes : HIL_Container<"nodes", [HasParent<"Graph">]> {
  let summary = "Nodes container.";
  let description = [{
    Container of `hil.node`
  }];
}

def HIL_NodeType : HIL_Op<"nodetype", [HasParent<"NodeTypes">]> {
  let summary = "Node type.";
  let description = [{
    The `hil.nodetype` operation for defining types of nodes
  }];

  let arguments = (ins
    StrAttr:$name,
    TypedArrayAttrBase<PortAttr, "input ports">:$commandArguments,
    TypedArrayAttrBase<PortAttr, "output ports">:$commandResults);
  let assemblyFormat = [{
    $name $commandArguments `=``>` $commandResults attr-dict
  }];
}

def HIL_Node : HIL_Op<"node", [HasParent<"Nodes">]> {
  let summary = "Node.";
  let description = [{
    The `hil.node` operation for defining nodes
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

def HIL_Chan : HIL_Op<"chan", [HasParent<"Chans">]> {
  let summary = "Channel.";
  let description = [{
    The `hil.chan` operation for defining channels
  }];

  let arguments = (ins
    StrAttr:$typeName,
    StrAttr:$varName,
    BindingAttr:$nodeFrom,
    BindingAttr:$nodeTo);
  let assemblyFormat = [{
    $typeName $varName qualified($nodeFrom) `=``=` qualified($nodeTo) attr-dict
  }];
}

def HIL_Con : HIL_Op<"con", [HasParent<"Cons">]> {
  let summary = "Connection.";
  let description = [{
    The `hil.con` operation for defining connections between graphs
  }];

  let arguments = (ins
    StrAttr:$typeName,
    StrAttr:$varName,
    StrAttr:$dirTypeName,
    BindingGraphAttr:$nodeFrom,
    BindingGraphAttr:$nodeTo);
  let assemblyFormat = [{
    $typeName $varName $dirTypeName qualified($nodeFrom) `=``=` qualified($nodeTo) attr-dict
  }];
}

#endif // HIL_OPS