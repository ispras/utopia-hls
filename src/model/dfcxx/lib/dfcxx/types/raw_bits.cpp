//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/types/raw_bits.h"

namespace dfcxx {

RawBitsType::RawBitsType(uint16_t bits) : bits(bits) {}

uint16_t RawBitsType::getTotalBits() const {
  return bits;
}

bool RawBitsType::operator==(const DFTypeImpl &rhs) const {
  if (rhs.isRawBits()) {
    const RawBitsType &casted = (const RawBitsType &) (rhs);
    return bits == casted.bits;
  }
  return false;
}

bool RawBitsType::isRawBits() const {
  return true;
}

} // namespace dfcxx
