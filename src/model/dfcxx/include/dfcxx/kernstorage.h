//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_KERNSTORAGE_H
#define DFCXX_KERNSTORAGE_H

#include "dfcxx/types/type.h"
#include "dfcxx/vars/var.h"

#include <functional>
#include <unordered_set>

namespace dfcxx {

class KernStorage {
using VarComp = std::function<bool(DFVariableImpl *, DFVariableImpl *)>;

private:
  std::unordered_set<DFTypeImpl *> types;
  std::unordered_set<DFVariableImpl *> variables;

public:
  DFTypeImpl *addType(DFTypeImpl *type);

  DFVariableImpl *addVariable(DFVariableImpl *var, VarComp cmp = {});

  void deleteVariable(DFVariableImpl *var);

  ~KernStorage();

  void transferFrom(KernStorage &&storage);
};

} // namespace dfcxx

#endif // DFCXX_KERNSTORAGE_H
