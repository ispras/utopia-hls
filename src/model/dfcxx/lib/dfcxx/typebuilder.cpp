//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/typebuilder.h"

namespace dfcxx {

DFTypeImpl *TypeBuilder::buildFixed(FixedType::SignMode mode,
                                    uint16_t intBits,
                                    uint16_t fracBits) {
  return new FixedType(mode, intBits, fracBits);
}

DFTypeImpl *TypeBuilder::buildBool() {
  return buildFixed(FixedType::SignMode::UNSIGNED, 1, 0);
}

DFTypeImpl *TypeBuilder::buildFloat(uint16_t expBits, uint16_t fracBits) {
  return new FloatType(expBits, fracBits);
}

DFTypeImpl *TypeBuilder::buildShiftedType(DFTypeImpl *type, uint16_t shift) {
  if (type->isFixed()) {
    FixedType *casted = (FixedType *) type;
    return buildFixed(
        casted->getSign(),
        casted->getIntBits() + shift,
        casted->getFracBits()
    );
  } else {
    FloatType *casted = (FloatType *) type;
    return buildFloat(
        casted->getExpBits(),
        casted->getFracBits() + shift
    );
  }
}

DFTypeImpl *TypeBuilder::buildRawBits(uint16_t bits) {
  return new RawBitsType(bits);
}

} // namespace dfcxx