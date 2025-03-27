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

Node::Node(DFVariableImpl *var) : var(var),
                                  type(OpType::NONE),
                                  data(NodeData {}),
                                  inputs(),
                                  outputs() {}

Node::Node(DFVariableImpl *var, OpType type, NodeData data) : var(var),
                                                              type(type),
                                                              data(data),
                                                              inputs(),
                                                              outputs() {}

Channel *Node::getConnection() {
  if (inputs.size() == 1 && inputs.front()->connect) {
    return inputs.front();
  }
  return nullptr;
}

bool Node::operator==(const dfcxx::Node &node) const {
  return var == node.var;
}

} // namespace dfcxx
