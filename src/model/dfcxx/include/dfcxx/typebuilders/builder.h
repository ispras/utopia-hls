//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_TYPE_BUILDER_H
#define DFCXX_TYPE_BUILDER_H

#include "dfcxx/types/types.h"

namespace dfcxx {

class TypeBuilder {
public:
  DFTypeImpl *buildFixed(FixedType::SignMode mode, uint16_t intBits,
                         uint16_t fracBits);

  DFTypeImpl *buildBool();

  DFTypeImpl *buildFloat(uint16_t expBits, uint16_t fracBits);

  DFTypeImpl *buildShiftedType(DFTypeImpl *type, uint16_t shift);

  DFTypeImpl *buildRawBits(uint16_t bits);
};

} // namespace dfcxx

#endif // DFCXX_TYPE_BUILDER_H
