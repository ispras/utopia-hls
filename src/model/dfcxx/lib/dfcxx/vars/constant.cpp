//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernmeta.h"
#include "dfcxx/vars/constant.h"

namespace dfcxx {

DFConstant::DFConstant(KernMeta &meta, DFTypeImpl *type, Value value) :
                       DFVariableImpl("", IODirection::NONE, meta),
                       type(*type), value(value) {
  if (this->type.isFixed()) {
    if (((FixedType &) this->type).isSigned()) {
      kind = TypeKind::INT;
    } else {
      kind = TypeKind::UINT;
    }
  } else {
    kind = TypeKind::FLOAT;
  }
}

DFVariableImpl *DFConstant::clone() const {
  return new DFConstant(meta, &type, value);
}

DFTypeImpl *DFConstant::getType() {
  return &type;
}

#define GENERIC_CONST_BINARY_OP(OP_TYPE, OP, VAR, RHS)                 \
DFVariableImpl *VAR;                                                   \
if (RHS.isConstant()) {                                                \
  Value val {};                                                        \
  DFConstant &casted = (DFConstant &) (RHS);                           \
  switch (kind) {                                                      \
    case INT:                                                          \
      val.int_ = value.int_ OP casted.value.int_;                      \
      break;                                                           \
    case UINT:                                                         \
      val.uint_ = value.uint_ OP casted.value.uint_;                   \
      break;                                                           \
    case FLOAT:                                                        \
      val.double_ = value.double_ OP casted.value.double_;             \
      break;                                                           \
  }                                                                    \
  VAR = meta.varBuilder.buildConstant(meta, &type, val);               \
  meta.storage.addVariable(VAR);                                       \
  meta.graph.addNode(VAR, OpType::CONST, NodeData {});                 \
  return VAR;                                                          \
}                                                                      \
VAR = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type); \
meta.storage.addVariable(VAR);                                         \
meta.graph.addNode(VAR, OP_TYPE, NodeData {});                         \
meta.graph.addChannel(this, VAR, 0, false);                            \
meta.graph.addChannel(&RHS, VAR, 1, false);                            \
return VAR;

DFVariableImpl *DFConstant::operator+(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::ADD, +, newVar, rhs)
}

DFVariableImpl *DFConstant::operator-(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::SUB, -, newVar, rhs)
}

DFVariableImpl *DFConstant::operator*(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::MUL, *, newVar, rhs)
}

DFVariableImpl *DFConstant::operator/(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::DIV, /, newVar, rhs)
}

#define GENERIC_CONST_BITWISE_OP(OP_TYPE, OP, VAR, RHS)                \
DFVariableImpl *VAR;                                                   \
if (RHS.isConstant()) {                                                \
  Value val {};                                                        \
  DFConstant &casted = (DFConstant &) (RHS);                           \
  val.uint_ = value.uint_ OP casted.value.uint_;                       \
  VAR = meta.varBuilder.buildConstant(meta, &type, val);               \
  meta.storage.addVariable(VAR);                                       \
  meta.graph.addNode(VAR, OpType::CONST, NodeData {});                 \
  return VAR;                                                          \
}                                                                      \
VAR = meta.varBuilder.buildStream("", IODirection::NONE, meta, &type); \
meta.storage.addVariable(VAR);                                         \
meta.graph.addNode(VAR, OP_TYPE, NodeData {});                         \
meta.graph.addChannel(this, VAR, 0, false);                            \
meta.graph.addChannel(&RHS, VAR, 1, false);                            \
return VAR;

DFVariableImpl *DFConstant::operator&(DFVariableImpl &rhs) {
  GENERIC_CONST_BITWISE_OP(OpType::AND, &, newVar, rhs)
}

DFVariableImpl *DFConstant::operator|(DFVariableImpl &rhs) {
  GENERIC_CONST_BITWISE_OP(OpType::OR, |, newVar, rhs)
}

DFVariableImpl *DFConstant::operator^(DFVariableImpl &rhs) {
  GENERIC_CONST_BITWISE_OP(OpType::XOR, ^, newVar, rhs)
}

DFVariableImpl *DFConstant::operator!() {
  DFVariableImpl *newVar;
  Value val {};
  val.uint_ = ~value.uint_;
  newVar = meta.varBuilder.buildConstant(meta, &type, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::CONST, NodeData {});
  return newVar;
}

DFVariableImpl *DFConstant::operator-() {
  DFVariableImpl *newVar;
  Value val {};
  switch (kind) {
    case INT:
      val.int_ = -value.int_;
      break;
    case UINT:
      val.uint_ = -value.uint_;
      break;
    case FLOAT:
      val.double_ = -value.double_;
      break;
  }
  newVar = meta.varBuilder.buildConstant(meta, &type, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::CONST, NodeData {});
  return newVar;
}

#define GENERIC_CONST_COMP_OP(OP_TYPE, OP, VAR, TYPE_VAR, RHS)             \
DFVariableImpl *VAR;                                                       \
if (RHS.isConstant()) {                                                    \
  Value val {};                                                            \
  DFConstant &casted = (DFConstant &) (RHS);                               \
  switch (kind) {                                                          \
    case INT:                                                              \
      val.int_ = value.int_ OP casted.value.int_;                          \
      break;                                                               \
    case UINT:                                                             \
      val.uint_ = value.uint_ OP casted.value.uint_;                       \
      break;                                                               \
    case FLOAT:                                                            \
      val.double_ = value.double_ OP casted.value.double_;                 \
      break;                                                               \
  }                                                                        \
  VAR = meta.varBuilder.buildConstant(meta, &type, val);                   \
  meta.storage.addVariable(VAR);                                           \
  meta.graph.addNode(VAR, OpType::CONST, NodeData {});                     \
  return VAR;                                                              \
}                                                                          \
DFTypeImpl *TYPE_VAR = meta.storage.addType(meta.typeBuilder.buildBool()); \
VAR = meta.varBuilder.buildStream("", IODirection::NONE, meta, TYPE_VAR);  \
meta.storage.addVariable(VAR);                                             \
meta.graph.addNode(VAR, OP_TYPE, NodeData {});                             \
meta.graph.addChannel(this, VAR, 0, false);                                \
meta.graph.addChannel(&RHS, VAR, 1, false);                                \
return VAR;

DFVariableImpl *DFConstant::operator<(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::LESS, <, newVar, newType, rhs)
}

DFVariableImpl *DFConstant::operator<=(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::LESSEQ, <=, newVar, newType, rhs)
}

DFVariableImpl *DFConstant::operator>(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::GREATER, >, newVar, newType, rhs)
}

DFVariableImpl *DFConstant::operator>=(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::GREATEREQ, >=, newVar, newType, rhs)
}

DFVariableImpl *DFConstant::operator==(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::EQ, ==, newVar, newType, rhs)
}

DFVariableImpl *DFConstant::operator!=(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::NEQ, !=, newVar, newType, rhs)
}

DFVariableImpl *DFConstant::operator<<(uint8_t bits) {
  DFVariableImpl *newVar;
  Value val {};
  val.uint_ = value.uint_ << bits;
  newVar = meta.varBuilder.buildConstant(meta, &type, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::CONST, NodeData {});
  return newVar;
}

DFVariableImpl *DFConstant::operator>>(uint8_t bits) {
  DFVariableImpl *newVar;
  Value val {};
  val.uint_ = value.uint_ >> bits;
  newVar = meta.varBuilder.buildConstant(meta, &type, val);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::CONST, NodeData {});
  return newVar;
}

DFConstant::TypeKind DFConstant::getKind() const {
  return kind;
}

int64_t DFConstant::getInt() const {
  return value.int_;
}

uint64_t DFConstant::getUInt() const {
  return value.uint_;
}

double DFConstant::getDouble() const {
  return value.double_;
}

bool DFConstant::isConstant() const {
  return true;
}

} // namespace dfcxx
