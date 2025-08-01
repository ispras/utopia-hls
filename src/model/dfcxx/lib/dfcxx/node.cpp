//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/channel.h"
#include "dfcxx/node.h"

namespace dfcxx {


ModuleInst::Port::Port(std::string name, Kind kind) :
    name(name), kind(kind) { }

ModuleInst::ModuleInst(std::string name, std::vector<Port> ports,
                       std::vector<ModuleParam> params) :
                       name(name), ports(ports), params(params)	{ }

Node::Node(DFVariableImpl *var, ModuleInst *inst) :
    var(var), inst(inst), type(OpType::NONE),
    data(NodeData {}), inputs(), outputs() {}

Node::Node(DFVariableImpl *var, ModuleInst *inst, OpType type, NodeData data) :
    var(var), inst(inst), type(type),
    data(data), inputs(), outputs() {}

Channel *Node::getConnection() {
  if (inputs.size() == 1 && inputs.front()->connect) {
    return inputs.front();
  }
  return nullptr;
}

bool Node::operator==(const dfcxx::Node &node) const {
  return var == node.var && inst == node.inst;
}

} // namespace dfcxx
