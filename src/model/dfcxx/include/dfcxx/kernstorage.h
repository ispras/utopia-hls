//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_KERNSTORAGE_H
#define DFCXX_KERNSTORAGE_H

#include "dfcxx/types/type.h"
#include "dfcxx/vars/var.h"

#include <unordered_set>

namespace dfcxx {

class KernStorage {
private:
  std::unordered_set<DFTypeImpl *> types;
  std::unordered_set<DFVariableImpl *> variables;

public:
  DFTypeImpl *addType(DFTypeImpl *type);

  DFVariableImpl *addVariable(DFVariableImpl *var);

  ~KernStorage();
};

} // namespace dfcxx

#endif // DFCXX_KERNSTORAGE_H
