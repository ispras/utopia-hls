#ifndef DFCXX_NODE_H
#define DFCXX_NODE_H

#include "dfcxx/vars/var.h"

namespace dfcxx {

enum OpType : uint8_t {
  OFFSET = 0,
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
  LESS_EQ,
  MORE,
  MORE_EQ,
  EQ,
  NEQ,
  SHL,
  SHR
};

union NodeData {
  int64_t offset;
  uint64_t muxId;
  uint8_t bitShift;
};

struct Node {
  DFVariableImpl *var;
  OpType type;
  NodeData data;


  Node(DFVariableImpl *var, OpType type, NodeData data);

  bool operator==(const Node &node) const;
};

} // namespace dfcxx

template <>
struct std::hash<dfcxx::Node> {
  size_t operator()(const dfcxx::Node &node) const noexcept {
    return std::hash<dfcxx::DFVariableImpl *>()(node.var);
  }
};

#endif // DFCXX_NODE_H
