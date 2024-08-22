//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/varbuilders/builder.h"
#include "dfcxx/vars/scalar.h"

namespace dfcxx {

DFScalar::DFScalar(const std::string &name, IODirection direction,
                   KernMeta &meta, DFTypeImpl *type) :
                   DFVariableImpl(name, direction, meta), type(*type) {}

DFTypeImpl *DFScalar::getType() {
  return &type;
}

DFVariableImpl *DFScalar::operator+(DFVariableImpl &rhs) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::ADD, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator-(DFVariableImpl &rhs) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SUB, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator*(DFVariableImpl &rhs) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::MUL, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator/(DFVariableImpl &rhs) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::DIV, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator&(DFVariableImpl &rhs) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::AND, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator|(DFVariableImpl &rhs) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::OR, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator^(DFVariableImpl &rhs) {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::XOR, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator!() {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::NOT, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator-() {
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, &type);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::NEG, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator<(DFVariableImpl &rhs) {
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::LESS, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator<=(DFVariableImpl &rhs) {
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::LESSEQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator>(DFVariableImpl &rhs) {
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::GREATER, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator>=(DFVariableImpl &rhs) {
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::GREATEREQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator==(DFVariableImpl &rhs) {
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::EQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator!=(DFVariableImpl &rhs) {
  DFTypeImpl *newType = meta.storage.addType(meta.typeBuilder.buildBool());
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::NEQ, NodeData{});
  meta.graph.addChannel(this, newVar, 0, false);
  meta.graph.addChannel(&rhs, newVar, 1, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator<<(uint8_t bits) {
  DFTypeImpl *newType = meta.storage.addType(
      meta.typeBuilder.buildShiftedType(type, bits));
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SHL, NodeData{.bitShift=bits});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

DFVariableImpl *DFScalar::operator>>(uint8_t bits) {
  DFTypeImpl *newType = meta.storage.addType(
      meta.typeBuilder.buildShiftedType(type, int8_t(bits) * -1));
  DFVariableImpl *newVar =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta.storage.addVariable(newVar);
  meta.graph.addNode(newVar, OpType::SHR, NodeData{.bitShift=bits});
  meta.graph.addChannel(this, newVar, 0, false);
  return newVar;
}

bool DFScalar::isScalar() const {
  return true;
}

} // namespace dfcxx
