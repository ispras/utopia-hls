//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_SCALAR_H
#define DFCXX_SCALAR_H

#include "dfcxx/vars/var.h"

namespace dfcxx {

class VarBuilder;

class DFScalar : public DFVariableImpl {
  friend VarBuilder;

private:
  DFScalar(const std::string &name, IODirection direction,
           KernMeta *meta, DFTypeImpl *type);

  DFVariableImpl *clone() const override;

public:
  ~DFScalar() override = default;

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

  bool isScalar() const override;
};

} // namespace dfcxx

#endif // DFCXX_SCALAR_H
