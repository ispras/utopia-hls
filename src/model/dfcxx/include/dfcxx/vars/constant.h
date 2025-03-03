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

class DFConstant : public DFVariableImpl {
  friend VarBuilder;

public:
  enum TypeKind : uint8_t {
    INT = 0,
    UINT,
    FLOAT
  };

  union Value {
    int64_t int_;
    uint64_t uint_;
    double double_;
  };

private:
  TypeKind kind;
  Value value;

  DFConstant(KernMeta *meta, DFTypeImpl *type, Value value);

  DFVariableImpl *clone() const override;

public:
  ~DFConstant() override = default;
  
  int64_t getInt() const;

  uint64_t getUInt() const;

  double getDouble() const;

  static TypeKind kindByType(DFTypeImpl *type);

  static DFVariableImpl *createOrUseConst(KernMeta *meta,
                                          DFTypeImpl *type,
                                          Value value);

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

  TypeKind getKind() const;

  bool isConstant() const override;
};

} // namespace dfcxx

#endif // DFCXX_CONST_H
