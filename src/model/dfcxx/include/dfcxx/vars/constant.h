//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CONST_H
#define DFCXX_CONST_H

#include "dfcxx/vars/var.h"

namespace dfcxx {

class VarBuilder;

class DFConstant : DFVariableImpl {
  friend VarBuilder;

  enum ConstantTypeKind : uint8_t {
    INT = 0,
    UINT,
    FLOAT
  };

  union ConstantValue {
    int64_t int_;
    uint64_t uint_;
    double double_;
  };

private:
  DFTypeImpl *type;
  ConstantTypeKind kind;
  ConstantValue value;

  DFConstant(KernMeta &meta, DFTypeImpl *type,
             ConstantTypeKind kind, ConstantValue value);

public:
  ~DFConstant() override = default;
  
  int64_t getInt() const;

  uint64_t getUInt() const;

  double getDouble() const;

  DFTypeImpl *getType() override;

  DFVariableImpl *operator+(DFVariableImpl &rhs) override;

  DFVariableImpl *operator-(DFVariableImpl &rhs) override;

  DFVariableImpl *operator*(DFVariableImpl &rhs) override;

  DFVariableImpl *operator/(DFVariableImpl &rhs) override;

  DFVariableImpl *operator&(DFVariableImpl &rhs) override;

  DFVariableImpl *operator|(DFVariableImpl &rhs) override;

  DFVariableImpl *operator^(DFVariableImpl &rhs) override;

  DFVariableImpl *operator!() override;

  DFVariableImpl *operator-() override;

  DFVariableImpl *operator<(DFVariableImpl &rhs) override;

  DFVariableImpl *operator<=(DFVariableImpl &rhs) override;

  DFVariableImpl *operator>(DFVariableImpl &rhs) override;

  DFVariableImpl *operator>=(DFVariableImpl &rhs) override;

  DFVariableImpl *operator==(DFVariableImpl &rhs) override;

  DFVariableImpl *operator!=(DFVariableImpl &rhs) override;

  DFVariableImpl *operator<<(uint8_t bits) override;

  DFVariableImpl *operator>>(uint8_t bits) override;

  ConstantTypeKind getKind() const;

  bool isConstant() const override;
};

} // namespace dfcxx

#endif // DFCXX_CONST_H
