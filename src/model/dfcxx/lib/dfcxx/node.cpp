//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/node.h"

namespace dfcxx {

Node::Node(DFVariableImpl *var) : var(var),
                                  type(OpType::NONE),
                                  data(NodeData {}) {}

Node::Node(DFVariableImpl *var, OpType type, NodeData data) : var(var),
                                                              type(type),
                                                              data(data) {}

bool Node::operator==(const dfcxx::Node &node) const {
  return var == node.var;
}

} // namespace dfcxx
