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
                   KernMeta *meta, DFTypeImpl *type) :
                   DFVariableImpl(name, direction, type, meta) {}

DFVariableImpl *DFStream::clone() const {
  return new DFStream(name, direction, meta, type);
}

#define GENERIC_STREAM_BINARY_OP(OP_TYPE)                            \
DFVariableImpl *newVar =                                             \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type); \
meta->graph.addNode(newVar, OP_TYPE, NodeData {});                   \
meta->graph.addChannel(this, newVar, 0, false);                      \
meta->graph.addChannel(&rhs, newVar, 1, false);                      \
meta->storage.addVariable(newVar);                                   \
return newVar

DFVariableImpl *DFStream::operator+(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::ADD);
}

DFVariableImpl *DFStream::operator-(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::SUB);
}

DFVariableImpl *DFStream::operator*(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::MUL);
}

DFVariableImpl *DFStream::operator/(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::DIV);
}

DFVariableImpl *DFStream::operator&(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::AND);
}

DFVariableImpl *DFStream::operator|(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::OR);
}

DFVariableImpl *DFStream::operator^(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::XOR);
}

#define GENERIC_STREAM_UNARY_OP(OP_TYPE)                             \
DFVariableImpl *newVar =                                             \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type); \
meta->graph.addNode(newVar, OP_TYPE, NodeData {});                   \
meta->graph.addChannel(this, newVar, 0, false);                      \
meta->storage.addVariable(newVar);                                   \
return newVar

DFVariableImpl *DFStream::operator!() {
  GENERIC_STREAM_UNARY_OP(OpType::NOT);
}

DFVariableImpl *DFStream::operator-() {
  GENERIC_STREAM_UNARY_OP(OpType::NEG);
}

#define GENERIC_STREAM_COMP_OP(OP_TYPE)                                   \
  DFTypeImpl *newType =                                                   \
      meta->storage.addType(meta->typeBuilder.buildBool());               \
  DFVariableImpl *newVar =                                                \
      meta->varBuilder.buildStream("", IODirection::NONE, meta, newType); \
  meta->storage.addVariable(newVar);                                      \
  meta->graph.addNode(newVar, OP_TYPE, NodeData {});                      \
  meta->graph.addChannel(this, newVar, 0, false);                         \
  meta->graph.addChannel(&rhs, newVar, 1, false);                         \
  return newVar

DFVariableImpl *DFStream::operator<(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESS);
}

DFVariableImpl *DFStream::operator<=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESSEQ);
}

DFVariableImpl *DFStream::operator>(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATER);
}

DFVariableImpl *DFStream::operator>=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATEREQ);
}

DFVariableImpl *DFStream::operator==(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::EQ);
}

DFVariableImpl *DFStream::operator!=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::NEQ);
}

DFVariableImpl *DFStream::operator<<(uint8_t bits) {
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta->storage.addVariable(newVar);
  meta->graph.addNode(newVar, OpType::SHL, NodeData {.bitShift=bits});
  meta->graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFStream::operator>>(uint8_t bits) {
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta->storage.addVariable(newVar);
  meta->graph.addNode(newVar, OpType::SHR, NodeData {.bitShift=bits});
  meta->graph.addChannel(this, newVar, 0, false);
  return newVar;
}

bool DFStream::isStream() const {
  return true;
}

} // namespace dfcxx
