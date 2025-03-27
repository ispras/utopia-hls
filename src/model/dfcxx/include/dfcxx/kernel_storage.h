//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_KERNEL_STORAGE_H
#define DFCXX_KERNEL_STORAGE_H

#include "dfcxx/types/type.h"
#include "dfcxx/vars/var.h"

#include <functional>
#include <unordered_set>

namespace dfcxx {

class KernelStorage {
using VarComp = std::function<bool(DFVariableImpl *, DFVariableImpl *)>;

public:
  // Types exist "outside" the scope of a singular kernel, so that
  // any kernel instance creates its variables with the same types as
  // the "parent" kernel.
  static std::unordered_set<DFTypeImpl *> types;

  std::unordered_set<DFVariableImpl *> variables;

public:
  DFTypeImpl *addType(DFTypeImpl *type);

  DFVariableImpl *addVariable(DFVariableImpl *var, VarComp cmp = {});

  void deleteVariable(DFVariableImpl *var);

  ~KernelStorage();

  void transferFrom(KernelStorage &&storage);
};

} // namespace dfcxx

#endif // DFCXX_KERNEL_STORAGE_H
