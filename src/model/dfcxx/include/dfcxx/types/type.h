//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_TYPE_H
#define DFCXX_TYPE_H

#include <cstdint>

namespace dfcxx {

class DFTypeImpl {
public:
  virtual ~DFTypeImpl() = default;

  virtual uint16_t getTotalBits() const = 0;

  virtual bool operator==(const DFTypeImpl &rhs) const = 0;

  bool operator!=(const DFTypeImpl &rhs) const;

  virtual bool isFixed() const;

  virtual bool isFloat() const;
};

class DFType {
private:
  DFTypeImpl *impl;

public:
  DFType(DFTypeImpl *impl);
  
  operator DFTypeImpl*() const;

  DFType(const DFType &) = default;

  DFTypeImpl *getImpl();

  uint16_t getTotalBits() const;

  bool operator==(const DFType &rhs) const;

  bool operator!=(const DFType &rhs) const;

  bool isFixed() const;

  bool isFloat() const;

  DFType &operator=(const DFType &type);
};

} // namespace dfcxx

#endif // DFCXX_TYPE_H
