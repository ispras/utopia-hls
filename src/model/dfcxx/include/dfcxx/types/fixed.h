//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_FIXED_H
#define DFCXX_FIXED_H

#include "dfcxx/types/type.h"

namespace dfcxx {

class TypeBuilder;

class FixedType : public DFTypeImpl {
  friend TypeBuilder;

public:
  enum SignMode {
    UNSIGNED = 0,
    SIGNED
  };

private:
  SignMode mode;
  uint16_t intBits;
  uint16_t fracBits;

  FixedType(SignMode mode, uint16_t intBits, uint16_t fracBits);

public:
  SignMode getSign() const;

  uint16_t getIntBits() const;

  uint16_t getFracBits() const;

  uint16_t getTotalBits() const override;

  ~FixedType() override = default;

  bool operator==(const DFTypeImpl &rhs) const override;

  bool isFixed() const override;

  bool isInt() const;

  bool isSigned() const;

  bool isUnsigned() const;

  bool isBool() const;
};

} // namespace dfcxx

#endif // DFCXX_FIXED_H
