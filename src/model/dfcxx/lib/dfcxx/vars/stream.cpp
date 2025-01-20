//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernmeta.h"
#include "dfcxx/vars/stream.h"

namespace dfcxx {

DFStream::DFStream(const std::string &name, IODirection direction,
                   KernMeta &meta, DFTypeImpl *type) :
                   DFVariableImpl(name, direction, meta), type(*type) {}

DFVariableImpl *DFStream::clone() const {
  return new DFStream(name, direction, meta, &type);
}

DFTypeImpl *DFStream::getType() {
  return &type;
}

#define GENERIC_STREAM_BINARY_OP(OP_TYPE, VAR, RHS)                  \
DFVariableImpl *VAR =                                                \
    meta.varBuilder.buildStream("", IODirection::NONE, meta, &type); \
meta.storage.addVariable(VAR);                                       \
meta.graph.addNode(VAR, OP_TYPE, NodeData {});                       \
meta.graph.addChannel(this, VAR, 0, false);                          \
meta.graph.addChannel(&RHS, VAR, 1, false);                          \
return VAR;

DFVariableImpl *DFStream::operator+(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::ADD, newVar, rhs)
}

DFVariableImpl *DFStream::operator-(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::SUB, newVar, rhs)
}

DFVariableImpl *DFStream::operator*(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::MUL, newVar, rhs)
}

DFVariableImpl *DFStream::operator/(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::DIV, newVar, rhs)
}

DFVariableImpl *DFStream::operator&(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::AND, newVar, rhs)
}

DFVariableImpl *DFStream::operator|(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::OR, newVar, rhs)
}

DFVariableImpl *DFStream::operator^(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::XOR, newVar, rhs)
}

#define GENERIC_STREAM_UNARY_OP(OP_TYPE, VAR)                        \
DFVariableImpl *VAR =                                                \
    meta.varBuilder.buildStream("", IODirection::NONE, meta, &type); \
meta.storage.addVariable(VAR);                                       \
meta.graph.addNode(VAR, OP_TYPE, NodeData {});                       \
meta.graph.addChannel(this, VAR, 0, false);                          \
return VAR;

DFVariableImpl *DFStream::operator!() {
  GENERIC_STREAM_UNARY_OP(OpType::NOT, newVar)
}

DFVariableImpl *DFStream::operator-() {
  GENERIC_STREAM_UNARY_OP(OpType::NEG, newVar)
}

#define GENERIC_STREAM_COMP_OP(OP_TYPE, VAR, TYPE_VAR, RHS)                \
DFTypeImpl *TYPE_VAR = meta.storage.addType(meta.typeBuilder.buildBool()); \
  DFVariableImpl *VAR =                                                    \
      meta.varBuilder.buildStream("", IODirection::NONE, meta, TYPE_VAR);  \
  meta.storage.addVariable(VAR);                                           \
  meta.graph.addNode(VAR, OP_TYPE, NodeData {});                           \
  meta.graph.addChannel(this, VAR, 0, false);                              \
  meta.graph.addChannel(&RHS, VAR, 1, false);                              \
  return VAR;

DFVariableImpl *DFStream::operator<(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESS, newVar, newType, rhs)
}

DFVariableImpl *DFStream::operator<=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESSEQ, newVar, newType, rhs)
}

DFVariableImpl *DFStream::operator>(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATER, newVar, newType, rhs)
}

DFVariableImpl *DFStream::operator>=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATEREQ, newVar, newType, rhs)
}

DFVariableImpl *DFStream::operator==(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::EQ, newVar, newType, rhs)
}

DFVariableImpl *DFStream::operator!=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::NEQ, newVar, newType, rhs)
}

DFVariableImpl *DFStream::operator<<(uint8_t bits) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SHL, NodeData {.bitShift=bits});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFStream::operator>>(uint8_t bits) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SHR, NodeData {.bitShift=bits});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

bool DFStream::isStream() const {
  return true;
}

} // namespace dfcxx
