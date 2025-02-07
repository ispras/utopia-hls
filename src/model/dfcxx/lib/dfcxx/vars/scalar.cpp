//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernmeta.h"
#include "dfcxx/vars/scalar.h"

namespace dfcxx {

DFScalar::DFScalar(const std::string &name, IODirection direction,
                   KernMeta *meta, DFTypeImpl *type) :
                   DFVariableImpl(name, direction, type, meta) {}

DFVariableImpl *DFScalar::clone() const {
  return new DFScalar(name, direction, meta, type);
}

#define GENERIC_STREAM_BINARY_OP(OP_TYPE, VAR, RHS)                   \
DFVariableImpl *VAR =                                                 \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type);  \
meta->storage.addVariable(VAR);                                       \
meta->graph.addNode(VAR, OP_TYPE, NodeData {});                       \
meta->graph.addChannel(this, VAR, 0, false);                          \
meta->graph.addChannel(&RHS, VAR, 1, false);                          \
return VAR;

DFVariableImpl *DFScalar::operator+(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::ADD, newVar, rhs)
}

DFVariableImpl *DFScalar::operator-(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::SUB, newVar, rhs)
}

DFVariableImpl *DFScalar::operator*(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::MUL, newVar, rhs)
}

DFVariableImpl *DFScalar::operator/(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::DIV, newVar, rhs)
}

DFVariableImpl *DFScalar::operator&(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::AND, newVar, rhs)
}

DFVariableImpl *DFScalar::operator|(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::OR, newVar, rhs)
}

DFVariableImpl *DFScalar::operator^(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::XOR, newVar, rhs)
}

#define GENERIC_STREAM_UNARY_OP(OP_TYPE, VAR)                        \
DFVariableImpl *VAR =                                                \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type); \
meta->storage.addVariable(VAR);                                      \
meta->graph.addNode(VAR, OP_TYPE, NodeData {});                      \
meta->graph.addChannel(this, VAR, 0, false);                         \
return VAR;

DFVariableImpl *DFScalar::operator!() {
  GENERIC_STREAM_UNARY_OP(OpType::NOT, newVar)
}

DFVariableImpl *DFScalar::operator-() {
  GENERIC_STREAM_UNARY_OP(OpType::NEG, newVar)
}

#define GENERIC_STREAM_COMP_OP(OP_TYPE, VAR, TYPE_VAR, RHS)                    \
  DFTypeImpl *TYPE_VAR = meta->storage.addType(meta->typeBuilder.buildBool()); \
  DFVariableImpl *VAR =                                                        \
      meta->varBuilder.buildStream("", IODirection::NONE, meta, TYPE_VAR);     \
  meta->storage.addVariable(VAR);                                              \
  meta->graph.addNode(VAR, OP_TYPE, NodeData {});                              \
  meta->graph.addChannel(this, VAR, 0, false);                                 \
  meta->graph.addChannel(&RHS, VAR, 1, false);                                 \
  return VAR;

DFVariableImpl *DFScalar::operator<(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESS, newVar, newType, rhs)
}

DFVariableImpl *DFScalar::operator<=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESSEQ, newVar, newType, rhs)
}

DFVariableImpl *DFScalar::operator>(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATER, newVar, newType, rhs)
}

DFVariableImpl *DFScalar::operator>=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATEREQ, newVar, newType, rhs)
}

DFVariableImpl *DFScalar::operator==(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::EQ, newVar, newType, rhs)
}

DFVariableImpl *DFScalar::operator!=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::NEQ, newVar, newType, rhs)
}

DFVariableImpl *DFScalar::operator<<(uint8_t bits) {
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta->storage.addVariable(newVar);
  meta->graph.addNode(newVar, OpType::SHL, NodeData {.bitShift=bits});
  meta->graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator>>(uint8_t bits) {
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta->storage.addVariable(newVar);
  meta->graph.addNode(newVar, OpType::SHR, NodeData {.bitShift=bits});
  meta->graph.addChannel(this, newVar, 0, false);
  return newVar;
}

bool DFScalar::isScalar() const {
  return true;
}

} // namespace dfcxx
