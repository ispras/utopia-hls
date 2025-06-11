//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernel_meta.h"
#include "dfcxx/vars/scalar.h"

namespace dfcxx {

DFScalar::DFScalar(const std::string &name, IODirection direction,
                   KernelMeta *meta, DFTypeImpl *type) :
                   DFVariableImpl(name, direction, type, meta) {}

DFVariableImpl *DFScalar::clone() const {
  return new DFScalar(name, direction, meta, type);
}

#define GENERIC_STREAM_BINARY_OP(OP_TYPE)                            \
DFVariableImpl *newVar =                                             \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type); \
meta->graph.addNode(newVar, nullptr, OP_TYPE, NodeData {});          \
meta->graph.addChannel(this, nullptr, newVar, nullptr, 0, false);    \
meta->graph.addChannel(&rhs, nullptr, newVar, nullptr, 1, false);    \
meta->storage.addVariable(newVar);                                   \
return newVar

DFVariableImpl *DFScalar::operator+(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::ADD);
}

DFVariableImpl *DFScalar::operator-(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::SUB);
}

DFVariableImpl *DFScalar::operator*(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::MUL);
}

DFVariableImpl *DFScalar::operator/(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::DIV);
}

DFVariableImpl *DFScalar::operator&(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::AND);
}

DFVariableImpl *DFScalar::operator|(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::OR);
}

DFVariableImpl *DFScalar::operator^(DFVariableImpl &rhs) {
  GENERIC_STREAM_BINARY_OP(OpType::XOR);
}

#define GENERIC_STREAM_UNARY_OP(OP_TYPE)                             \
DFVariableImpl *newVar =                                             \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type); \
meta->storage.addVariable(newVar);                                   \
meta->graph.addNode(newVar, nullptr, OP_TYPE, NodeData {});          \
meta->graph.addChannel(this, nullptr, newVar, nullptr, 0, false);    \
return newVar

DFVariableImpl *DFScalar::operator!() {
  GENERIC_STREAM_UNARY_OP(OpType::NOT);
}

DFVariableImpl *DFScalar::operator-() {
  GENERIC_STREAM_UNARY_OP(OpType::NEG);
}

#define GENERIC_STREAM_COMP_OP(OP_TYPE)                                   \
  DFTypeImpl *newType =                                                   \
      meta->storage.addType(meta->typeBuilder.buildBool());               \
  DFVariableImpl *newVar =                                                \
      meta->varBuilder.buildStream("", IODirection::NONE, meta, newType); \
  meta->storage.addVariable(newVar);                                      \
  meta->graph.addNode(newVar, nullptr, OP_TYPE, NodeData {});             \
  meta->graph.addChannel(this, nullptr, newVar, nullptr, 0, false);       \
  meta->graph.addChannel(&rhs, nullptr, newVar, nullptr, 1, false);       \
  return newVar

DFVariableImpl *DFScalar::operator<(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESS);
}

DFVariableImpl *DFScalar::operator<=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::LESSEQ);
}

DFVariableImpl *DFScalar::operator>(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATER);
}

DFVariableImpl *DFScalar::operator>=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::GREATEREQ);
}

DFVariableImpl *DFScalar::operator==(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::EQ);
}

DFVariableImpl *DFScalar::operator!=(DFVariableImpl &rhs) {
  GENERIC_STREAM_COMP_OP(OpType::NEQ);
}

DFVariableImpl *DFScalar::operator<<(uint8_t bits) {
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta->storage.addVariable(newVar);
  meta->graph.addNode(newVar, nullptr, OpType::SHL, NodeData {.bitShift=bits});
  meta->graph.addChannel(this, nullptr, newVar, nullptr, 0, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator>>(uint8_t bits) {
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta->storage.addVariable(newVar);
  meta->graph.addNode(newVar, nullptr, OpType::SHR, NodeData {.bitShift=bits});
  meta->graph.addChannel(this, nullptr, newVar, nullptr, 0, false);
  return newVar;
}

bool DFScalar::isScalar() const {
  return true;
}

} // namespace dfcxx
