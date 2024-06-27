//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/types/float.h"

namespace dfcxx {

FloatType::FloatType(uint8_t expBits, uint8_t fracBits) : expBits(expBits),
                                                          fracBits(fracBits) {}

uint8_t FloatType::getExpBits() const {
  return expBits;
}

uint8_t FloatType::getFracBits() const {
  return fracBits;
}

uint16_t FloatType::getTotalBits() const {
  return uint16_t(expBits) + fracBits;
}

bool FloatType::operator==(const DFTypeImpl &rhs) const {
  const FloatType &casted = (const FloatType &) (rhs);
  return expBits == casted.expBits && fracBits == casted.fracBits;
}

bool FloatType::isFloat() const {
  return true;
}

} // namespace dfcxx
