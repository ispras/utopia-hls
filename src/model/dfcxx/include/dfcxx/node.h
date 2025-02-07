//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_NODE_H
#define DFCXX_NODE_H

#include "dfcxx/vars/var.h"

namespace dfcxx {

enum OpType : uint8_t {
  NONE = 0, // Is not allowed in a fully constructed kernel.
  OFFSET,
  IN,
  OUT,
  CONST,
  MUX,
  ADD,
  SUB,
  MUL,
  DIV,
  AND,
  OR,
  XOR,
  NOT,
  NEG,
  LESS,
  LESSEQ,
  GREATER,
  GREATEREQ,
  EQ,
  NEQ,
  CAST,
  SHL,
  SHR,
  BITS,
  CAT
};

union NodeData {
  int64_t offset;
  uint64_t muxId;
  uint8_t bitShift;
  struct {
    uint8_t left;
    uint8_t right;
  } bitsRange;
};

struct Node {
  DFVariableImpl *var;
  OpType type;
  NodeData data;
  
  Node() = default;
  Node(DFVariableImpl *var, OpType type, NodeData data);

  bool operator==(const Node &node) const;
  bool operator!=(const Node &node) const { return !(*this == node); }
};

} // namespace dfcxx

template <>
struct std::hash<dfcxx::Node> {
  size_t operator()(const dfcxx::Node &node) const noexcept {
    return std::hash<dfcxx::DFVariableImpl *>()(node.var);
  }
};

#endif // DFCXX_NODE_H
