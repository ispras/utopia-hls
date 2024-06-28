//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_SCALAR_H
#define DFCXX_SCALAR_H

#include "dfcxx/vars/var.h"

namespace dfcxx {

class VarBuilder;
class DFCIRBuilder;

class DFScalar : DFVariableImpl {
  friend VarBuilder;
  friend DFCIRBuilder;

private:
  DFTypeImpl &type;

  DFScalar(const std::string &name, IODirection direction,
           GraphHelper &helper, DFTypeImpl &type);

public:
  ~DFScalar() override = default;

protected:
  DFTypeImpl &getType() override;

  DFVariableImpl &operator+(DFVariableImpl &rhs) override;

  DFVariableImpl &operator-(DFVariableImpl &rhs) override;

  DFVariableImpl &operator*(DFVariableImpl &rhs) override;

  DFVariableImpl &operator/(DFVariableImpl &rhs) override;

  DFVariableImpl &operator&(DFVariableImpl &rhs) override;

  DFVariableImpl &operator|(DFVariableImpl &rhs) override;

  DFVariableImpl &operator^(DFVariableImpl &rhs) override;

  DFVariableImpl &operator!() override;

  DFVariableImpl &operator-() override;

  DFVariableImpl &operator<(DFVariableImpl &rhs) override;

  DFVariableImpl &operator<=(DFVariableImpl &rhs) override;

  DFVariableImpl &operator>(DFVariableImpl &rhs) override;

  DFVariableImpl &operator>=(DFVariableImpl &rhs) override;

  DFVariableImpl &operator==(DFVariableImpl &rhs) override;

  DFVariableImpl &operator!=(DFVariableImpl &rhs) override;

  DFVariableImpl &operator<<(uint8_t bits) override;

  DFVariableImpl &operator>>(uint8_t bits) override;

  bool isScalar() const override;
};

} // namespace dfcxx

#endif // DFCXX_SCALAR_H
