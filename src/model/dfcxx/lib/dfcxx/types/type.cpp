//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/types/type.h"

namespace dfcxx {

bool DFTypeImpl::operator!=(const DFTypeImpl &rhs) const {
  return !(*this == rhs);
}

bool DFTypeImpl::isFixed() const {
  return false;
}

bool DFTypeImpl::isFloat() const {
  return false;
}

bool DFTypeImpl::isRawBits() const {
  return false;
}

DFType::DFType(DFTypeImpl *impl) : impl(impl) {}

DFType::operator DFTypeImpl*() const {
  return impl;
}

DFTypeImpl *DFType::getImpl() {
  return impl;
}

uint16_t DFType::getTotalBits() const {
  return impl->getTotalBits();
}

bool DFType::operator==(const DFType &rhs) const {
  return impl->operator==(*(rhs.impl));
}

bool DFType::operator!=(const DFType &rhs) const {
  return !(*this == rhs);
}

bool DFType::isFixed() const {
  return impl->isFixed();
}

bool DFType::isFloat() const {
  return impl->isFloat();
}

bool DFType::isRawBits() const {
  return impl->isRawBits();
}

DFType &DFType::operator=(const DFType &type) {
  impl = type.impl;
  return *this;
}

} // namespace dfcxx
