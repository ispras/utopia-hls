//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernel_meta.h"
#include "dfcxx/vars/var.h"

#include <sstream>
#include <stdexcept>

namespace dfcxx {

DFVariableImpl::DFVariableImpl(const std::string &name,
                               IODirection direction,
                               DFTypeImpl *type,
                               KernelMeta *meta) : name(name),
                                                 direction(direction),
                                                 type(type),
                                                 meta(meta) {}

std::string_view DFVariableImpl::getName() const {
  return name;
}

void DFVariableImpl::resetName() {
  name.clear();
}

DFVariableImpl::IODirection DFVariableImpl::getDirection() const {
  return direction;
}

KernelMeta *DFVariableImpl::getMeta() const {
  return meta;
}

void DFVariableImpl::setMeta(KernelMeta *meta) {
  this->meta = meta;
}

DFTypeImpl *DFVariableImpl::getType() {
  return type;
}

const DFTypeImpl *DFVariableImpl::getType() const {
  return type;
}

uint16_t DFVariableImpl::getTotalBits() const {
  return getType()->getTotalBits();
}

bool DFVariableImpl::isStream() const {
  return false;
}

bool DFVariableImpl::isScalar() const {
  return false;
}

bool DFVariableImpl::isConstant() const {
  return false;
}

void DFVariableImpl::connect(DFVariableImpl *connectee) {
  meta->graph.addChannel(connectee, this, 0, true);
}

DFVariableImpl *DFVariableImpl::operator()(uint16_t first, uint16_t second) {
  uint16_t left = (first > second) ? first : second;
  uint16_t right = (first > second) ? second : first;
  uint16_t total = getTotalBits();
  if (total <= left) {
    std::stringstream ss;
    ss << "Invalid range [" << left << ", " << right;
    ss << "] for " << total << " bits.";
    throw new std::runtime_error(ss.str());
  }
  DFTypeImpl *newType = meta->storage.addType(
      meta->typeBuilder.buildRawBits(left - right + 1));
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, newType);
  NodeData data = {.bitsRange={.left=left, .right=right}};
  meta->graph.addNode(newVar, OpType::BITS, data);
  meta->graph.addChannel(this, newVar, 0, false);
  meta->storage.addVariable(newVar);
  return newVar;
}

DFVariableImpl *DFVariableImpl::cat(DFVariableImpl &rhs) {
  uint16_t total = getTotalBits() + rhs.getTotalBits();
  DFTypeImpl *newType = meta->storage.addType(
      meta->typeBuilder.buildRawBits(total));
  DFVariableImpl *newVar =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, newType);
  meta->graph.addNode(newVar, OpType::CAT, NodeData{});
  meta->graph.addChannel(this, newVar, 0, false);
  meta->graph.addChannel(&rhs, newVar, 1, false);
  meta->storage.addVariable(newVar);
  return newVar;
}

DFVariableImpl *DFVariableImpl::cast(DFTypeImpl *type) {
  if (type == getType()) {
    return this;
  }

  DFVariableImpl *var =
      meta->varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta->graph.addNode(var, OpType::CAST, NodeData {});
  meta->graph.addChannel(this, var, 0, false);
  meta->storage.addVariable(var);
  return var;
}

DFVariable::DFVariable(DFVariableImpl *impl) : impl(impl) {}

DFVariable::operator DFVariableImpl*() const {
  return impl;
}

DFVariableImpl *DFVariable::getImpl() const {
  return impl;
}

std::string_view DFVariable::getName() const {
  return impl->getName();
}

DFVariableImpl::IODirection DFVariable::getDirection() const {
  return impl->getDirection();
}

DFType DFVariable::getType() const {
  return DFType(impl->getType());
}

uint16_t DFVariable::getTotalBits() const {
  return impl->getTotalBits();
}

DFVariable DFVariable::operator+(const DFVariable &rhs) {
  return DFVariable(impl->operator+(*(rhs.impl)));
}

DFVariable DFVariable::operator-(const DFVariable &rhs) {
  return DFVariable(impl->operator-(*(rhs.impl)));
}

DFVariable DFVariable::operator*(const DFVariable &rhs) {
  return DFVariable(impl->operator*(*(rhs.impl)));
}

DFVariable DFVariable::operator/(const DFVariable &rhs) {
  return DFVariable(impl->operator/(*(rhs.impl)));
}

DFVariable DFVariable::operator&(const DFVariable &rhs) {
  return DFVariable(impl->operator&(*(rhs.impl)));
}

DFVariable DFVariable::operator|(const DFVariable &rhs) {
  return DFVariable(impl->operator|(*(rhs.impl)));
}

DFVariable DFVariable::operator^(const DFVariable &rhs) {
  return DFVariable(impl->operator^(*(rhs.impl)));
}

DFVariable DFVariable::operator!() {
  return DFVariable(impl->operator!());
}

DFVariable DFVariable::operator-() {
  return DFVariable(impl->operator-());
}

DFVariable DFVariable::operator<(const DFVariable &rhs) {
  return DFVariable(impl->operator<(*(rhs.impl)));
}

DFVariable DFVariable::operator<=(const DFVariable &rhs) {
  return DFVariable(impl->operator<=(*(rhs.impl)));
}

DFVariable DFVariable::operator>(const DFVariable &rhs) {
  return DFVariable(impl->operator>(*(rhs.impl)));
}

DFVariable DFVariable::operator>=(const DFVariable &rhs) {
  return DFVariable(impl->operator>=(*(rhs.impl)));
}

DFVariable DFVariable::operator==(const DFVariable &rhs) {
  return DFVariable(impl->operator==(*(rhs.impl)));
}

DFVariable DFVariable::operator!=(const DFVariable &rhs) {
  return DFVariable(impl->operator!=(*(rhs.impl)));
}

DFVariable DFVariable::operator<<(uint8_t bits) {
  return DFVariable(impl->operator<<(bits));
}

DFVariable DFVariable::operator>>(uint8_t bits) {
  return DFVariable(impl->operator>>(bits));
}

bool DFVariable::isStream() const {
  return impl->isStream();
}

bool DFVariable::isScalar() const {
  return impl->isScalar();
}

bool DFVariable::isConstant() const {
  return impl->isConstant();
}

DFVariable DFVariable::cast(const DFType &type) {
  return impl->cast(type);
}

void DFVariable::connect(const DFVariable &connectee) {
  impl->connect(connectee.impl);
}

DFVariable DFVariable::operator()(uint16_t first, uint16_t second) {
  return DFVariable(impl->operator()(first, second));
}

DFVariable DFVariable::cat(const DFVariable &rhs) {
  return DFVariable(impl->cat(*(rhs.impl)));
}

} // namespace dfcxx
