//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_RAW_BITS_H
#define DFCXX_RAW_BITS_H

#include "dfcxx/types/type.h"

namespace dfcxx {

class TypeBuilder;

class RawBitsType : public DFTypeImpl {
  friend TypeBuilder;

private:
  uint16_t bits;

  RawBitsType(uint16_t bits);

public:
  uint16_t getTotalBits() const override;

  ~RawBitsType() override = default;

  bool isRawBits() const override;

  bool operator==(const DFTypeImpl &rhs) const override;
};

} // namespace dfcxx

#endif // DFCXX_RAW_BITS_H
