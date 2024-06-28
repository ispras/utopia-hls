//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_FLOAT_H
#define DFCXX_FLOAT_H

#include "dfcxx/types/type.h"

namespace dfcxx {

class TypeBuilder;

class FloatType : DFTypeImpl {
  friend TypeBuilder;

private:
  uint8_t expBits;
  uint8_t fracBits;

  FloatType(uint8_t expBits, uint8_t fracBits);

public:
  uint8_t getExpBits() const;

  uint8_t getFracBits() const;

  uint16_t getTotalBits() const override;

  ~FloatType() override = default;

  bool operator==(const DFTypeImpl &rhs) const override;

  bool isFloat() const override;
};

} // namespace dfcxx

#endif // DFCXX_FLOAT_H
