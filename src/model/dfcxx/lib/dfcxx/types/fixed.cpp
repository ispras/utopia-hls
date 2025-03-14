//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/types/fixed.h"

namespace dfcxx {

FixedType::FixedType(SignMode mode, uint16_t intBits,
                     uint16_t fracBits) : mode(mode), intBits(intBits),
                                          fracBits(fracBits) {}

FixedType::SignMode FixedType::getSign() const {
  return mode;
}

uint16_t FixedType::getIntBits() const {
  return intBits;
}

uint16_t FixedType::getFracBits() const {
  return fracBits;
}

uint16_t FixedType::getTotalBits() const {
  return uint16_t(isSigned()) + intBits + fracBits;
}

bool FixedType::operator==(const DFTypeImpl &rhs) const {
  if (rhs.isFixed()) {
    const FixedType &casted = (const FixedType &) (rhs);
    return mode == casted.mode &&
           intBits == casted.intBits &&
           fracBits == casted.fracBits;
  }
  return false;
}

bool FixedType::isFixed() const {
  return true;
}

bool FixedType::isInt() const {
  return fracBits == 0;
}

bool FixedType::isSigned() const {
  return mode == SignMode::SIGNED;
}

bool FixedType::isUnsigned() const {
  return mode == SignMode::UNSIGNED;
}

bool FixedType::isBool() const {
  return mode == SignMode::UNSIGNED && intBits == 1 && fracBits == 0;
}

} // namespace dfcxx
