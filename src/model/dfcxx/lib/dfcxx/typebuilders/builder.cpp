//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/typebuilders/builder.h"

namespace dfcxx {

DFTypeImpl *TypeBuilder::buildFixed(FixedType::SignMode mode,
                                    uint8_t intBits,
                                    uint8_t fracBits) {
  return new FixedType(mode, intBits, fracBits);
}

DFTypeImpl *TypeBuilder::buildBool() {
  return buildFixed(FixedType::SignMode::UNSIGNED, 1, 0);
}

DFTypeImpl *TypeBuilder::buildFloat(uint8_t expBits, uint8_t fracBits) {
  return new FloatType(expBits, fracBits);
}

DFTypeImpl *TypeBuilder::buildShiftedType(DFTypeImpl *type, int8_t shift) {
  if (type->isFixed()) {
    FixedType *casted = (FixedType *) type;
    return buildFixed(
        casted->getSign(),
        uint8_t(int16_t(casted->getIntBits()) + shift),
        casted->getFracBits());
  } else {
    FloatType *casted = (FloatType *) type;
    return buildFloat(
        casted->getExpBits(),
        uint8_t(uint16_t(casted->getFracBits()) + shift));
  }
}

DFTypeImpl *TypeBuilder::buildRawBits(uint16_t bits) {
  return new RawBitsType(bits);
}

} // namespace dfcxx