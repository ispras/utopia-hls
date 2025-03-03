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

DFConstant::DFConstant(KernMeta *meta, DFTypeImpl *type, Value value) :
                       DFVariableImpl("", IODirection::NONE, type, meta),
                       kind(kindByType(type)),
                       value(value) { }

DFVariableImpl *DFConstant::clone() const {
  return new DFConstant(meta, type, value);
}

DFConstant::TypeKind DFConstant::kindByType(DFTypeImpl *type) {
  if (type->isFixed()) {
    if (((FixedType *) type)->isSigned()) {
      return TypeKind::INT;
    } else {
      return TypeKind::UINT;
    }
  } else {
    return TypeKind::FLOAT;
  }
}

DFVariableImpl *DFConstant::createOrUseConst(KernMeta *meta,
                                             DFTypeImpl *type,
                                             Value value) {
  TypeKind kind = kindByType(type);
  Node *found = meta->graph.findStartNodeIf([kind, value] (Node *node) {
    if (node->type != OpType::CONST || !node->var->isConstant()) {
      return false;
    }
    DFConstant *casted = (DFConstant *)node->var;
    return casted->getKind() == kind && casted->getUInt() == value.uint_;
  });

  if (found) { return found->var; }

  auto *var = meta->varBuilder.buildConstant(meta, type,
                                             value);
  meta->graph.addNode(var, OpType::CONST, NodeData {});
  meta->storage.addVariable(var);
  return var;
}

#define GENERIC_CONST_BINARY_OP(OP_TYPE, OP)                         \
DFVariableImpl *newVar;                                              \
if (rhs.isConstant()) {                                              \
  Value val {};                                                      \
  DFConstant &casted = (DFConstant &) (rhs);                         \
  switch (kind) {                                                    \
    case INT:                                                        \
      val.int_ = value.int_ OP casted.value.int_;                    \
      break;                                                         \
    case UINT:                                                       \
      val.uint_ = value.uint_ OP casted.value.uint_;                 \
      break;                                                         \
    case FLOAT:                                                      \
      val.double_ = value.double_ OP casted.value.double_;           \
      break;                                                         \
  }                                                                  \
  return createOrUseConst(meta, type, val);                          \
}                                                                    \
newVar =                                                             \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type); \
meta->graph.addNode(newVar, OP_TYPE, NodeData {});                   \
meta->graph.addChannel(this, newVar, 0, false);                      \
meta->graph.addChannel(&rhs, newVar, 1, false);                      \
meta->storage.addVariable(newVar);                                   \
return newVar

DFVariableImpl *DFConstant::operator+(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::ADD, +);
}

DFVariableImpl *DFConstant::operator-(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::SUB, -);
}

DFVariableImpl *DFConstant::operator*(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::MUL, *);
}

DFVariableImpl *DFConstant::operator/(DFVariableImpl &rhs) {
  GENERIC_CONST_BINARY_OP(OpType::DIV, /);
}

#define GENERIC_CONST_BITWISE_OP(OP_TYPE, OP)                        \
DFVariableImpl *newVar;                                              \
if (rhs.isConstant()) {                                              \
  Value val {};                                                      \
  DFConstant &casted = (DFConstant &) (rhs);                         \
  val.uint_ = value.uint_ OP casted.value.uint_;                     \
  return createOrUseConst(meta, type, val);                          \
}                                                                    \
newVar =                                                             \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, type); \
meta->graph.addNode(newVar, OP_TYPE, NodeData {});                   \
meta->graph.addChannel(this, newVar, 0, false);                      \
meta->graph.addChannel(&rhs, newVar, 1, false);                      \
meta->storage.addVariable(newVar);                                   \
return newVar

DFVariableImpl *DFConstant::operator&(DFVariableImpl &rhs) {
  GENERIC_CONST_BITWISE_OP(OpType::AND, &);
}

DFVariableImpl *DFConstant::operator|(DFVariableImpl &rhs) {
  GENERIC_CONST_BITWISE_OP(OpType::OR, |);
}

DFVariableImpl *DFConstant::operator^(DFVariableImpl &rhs) {
  GENERIC_CONST_BITWISE_OP(OpType::XOR, ^);
}

DFVariableImpl *DFConstant::operator!() {
  Value val {};
  val.uint_ = ~value.uint_;
  return createOrUseConst(meta, type, val);
}

DFVariableImpl *DFConstant::operator-() {
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
  return createOrUseConst(meta, type, val);
}

#define GENERIC_CONST_COMP_OP(OP_TYPE, OP)                              \
DFVariableImpl *newVar;                                                 \
if (rhs.isConstant()) {                                                 \
  Value val {};                                                         \
  DFConstant &casted = (DFConstant &) (rhs);                            \
  switch (kind) {                                                       \
    case INT:                                                           \
      val.int_ = value.int_ OP casted.value.int_;                       \
      break;                                                            \
    case UINT:                                                          \
      val.uint_ = value.uint_ OP casted.value.uint_;                    \
      break;                                                            \
    case FLOAT:                                                         \
      val.double_ = value.double_ OP casted.value.double_;              \
      break;                                                            \
  }                                                                     \
  return createOrUseConst(meta, type, val);                             \
}                                                                       \
DFTypeImpl *newType =                                                   \
    meta->storage.addType(meta->typeBuilder.buildBool());               \
newVar =                                                                \
    meta->varBuilder.buildStream("", IODirection::NONE, meta, newType); \
meta->graph.addNode(newVar, OP_TYPE, NodeData {});                      \
meta->graph.addChannel(this, newVar, 0, false);                         \
meta->graph.addChannel(&rhs, newVar, 1, false);                         \
meta->storage.addVariable(newVar);                                      \
return newVar

DFVariableImpl *DFConstant::operator<(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::LESS, <);
}

DFVariableImpl *DFConstant::operator<=(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::LESSEQ, <=);
}

DFVariableImpl *DFConstant::operator>(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::GREATER, >);
}

DFVariableImpl *DFConstant::operator>=(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::GREATEREQ, >=);
}

DFVariableImpl *DFConstant::operator==(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::EQ, ==);
}

DFVariableImpl *DFConstant::operator!=(DFVariableImpl &rhs) {
  GENERIC_CONST_COMP_OP(OpType::NEQ, !=);
}

DFVariableImpl *DFConstant::operator<<(uint8_t bits) {
  Value val {};
  val.uint_ = value.uint_ << bits;
  return createOrUseConst(meta, type, val);
}

DFVariableImpl *DFConstant::operator>>(uint8_t bits) {
  Value val {};
  val.uint_ = value.uint_ >> bits;
  return createOrUseConst(meta, type, val);
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
