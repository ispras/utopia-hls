//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernmeta.h"
#include "dfcxx/vars/var.h"

namespace dfcxx {

DFVariableImpl::DFVariableImpl(const std::string &name,
                               IODirection direction,
                               KernMeta &meta) : name(name),
                                                 direction(direction),
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

const KernMeta &DFVariableImpl::getMeta() const {
  return meta;
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
  meta.graph.addChannel(connectee, this, 0, true);
}

DFVariableImpl *DFVariableImpl::cast(DFTypeImpl *type) {
  DFVariableImpl *var =
      meta.varBuilder.buildStream("", IODirection::NONE, meta, type);
  meta.storage.addVariable(var);
  meta.graph.addNode(var, OpType::CAST, NodeData {});
  meta.graph.addChannel(this, var, 0, false);
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

DFVariable &DFVariable::operator=(const DFVariable &var) {
  impl = var.impl;
  return *this;
}

} // namespace dfcxx
