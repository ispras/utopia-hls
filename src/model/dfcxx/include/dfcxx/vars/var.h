//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_VAR_H
#define DFCXX_VAR_H

#include "dfcxx/types/types.h"

#include <string_view>
#include <string>

namespace dfcxx {

class VarBuilder;
struct KernMeta; // Forward declaration to omit cyclic dependency.

class DFVariableImpl {
  friend VarBuilder;

public:
  enum IODirection {
    NONE = 0,
    INPUT,
    OUTPUT
  };

protected:
  std::string name;
  IODirection direction;
  DFTypeImpl *type;
  KernMeta *meta;

  virtual DFVariableImpl *clone() const = 0;

public:
  DFVariableImpl(const std::string &name, IODirection direction,
                 DFTypeImpl *type, KernMeta *meta);

  virtual ~DFVariableImpl() = default;

  virtual bool isStream() const;

  virtual bool isScalar() const;

  virtual bool isConstant() const;

  std::string_view getName() const;

  void resetName();

  IODirection getDirection() const;

  const KernMeta &getMeta() const;

  DFTypeImpl *getType();

  const DFTypeImpl *getType() const;

  uint16_t getTotalBits() const;

  virtual DFVariableImpl *operator+(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator-(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator*(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator/(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator&(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator|(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator^(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator!() = 0;

  virtual DFVariableImpl *operator-() = 0;

  virtual DFVariableImpl *operator<(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator<=(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator>(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator>=(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator==(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator!=(DFVariableImpl &rhs) = 0;

  virtual DFVariableImpl *operator<<(uint8_t bits) = 0;

  virtual DFVariableImpl *operator>>(uint8_t bits) = 0;

  void connect(DFVariableImpl *connectee);

  DFVariableImpl *cast(DFTypeImpl *type);

  virtual DFVariableImpl *operator()(uint8_t first, uint8_t second);

  virtual DFVariableImpl *cat(DFVariableImpl &rhs);
};

class DFVariable {
private:
  DFVariableImpl *impl;

public:
  DFVariable(DFVariableImpl *impl);

  operator DFVariableImpl*() const;

  DFVariable(const DFVariable &) = default;

  DFVariableImpl *getImpl() const;

  std::string_view getName() const;

  DFVariableImpl::IODirection getDirection() const;

  const KernMeta &getMeta() const;

  DFType getType() const;

  uint16_t getTotalBits() const;

  DFVariable operator+(const DFVariable &rhs);

  DFVariable operator-(const DFVariable &rhs);

  DFVariable operator*(const DFVariable &rhs);

  DFVariable operator/(const DFVariable &rhs);

  DFVariable operator&(const DFVariable &rhs);

  DFVariable operator|(const DFVariable &rhs);

  DFVariable operator^(const DFVariable &rhs);

  DFVariable operator!();

  DFVariable operator-();

  DFVariable operator<(const DFVariable &rhs);

  DFVariable operator<=(const DFVariable &rhs);

  DFVariable operator>(const DFVariable &rhs);

  DFVariable operator>=(const DFVariable &rhs);

  DFVariable operator==(const DFVariable &rhs);

  DFVariable operator!=(const DFVariable &rhs);

  DFVariable operator<<(uint8_t bits);

  DFVariable operator>>(uint8_t bits);

  bool isStream() const;

  bool isScalar() const;

  bool isConstant() const;

  void connect(const DFVariable &connectee);

  DFVariable cast(const DFType &type);

  DFVariable operator()(uint8_t first, uint8_t second);

  DFVariable cat(const DFVariable &rhs);

  DFVariable &operator=(const DFVariable &var);
};

} // namespace dfcxx

#endif // DFCXX_VAR_H
