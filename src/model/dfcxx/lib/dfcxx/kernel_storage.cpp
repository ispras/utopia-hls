//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernel_storage.h"

#include <algorithm>

namespace dfcxx {

dfcxx::DFTypeImpl *KernelStorage::addType(dfcxx::DFTypeImpl *type) {
  std::unordered_set<DFTypeImpl *> types;

  auto found = std::find_if(types.begin(), types.end(),
                            [&] (dfcxx::DFTypeImpl *t) {
                              return t->operator==(*type);
                            });
  if (found != types.end()) {
    delete type;
    return *found;
  }
  return *(types.insert(type).first);
}

DFVariableImpl *KernelStorage::addVariable(DFVariableImpl *var,  VarComp cmp) {
  if (cmp) {
    for (DFVariableImpl *iter: variables) {
      if (cmp(var, iter)) {
        delete var;
        return iter;
      }
    }
  }
  return *(variables.insert(var).first);
}

void KernelStorage::deleteVariable(DFVariableImpl *var) {
  variables.erase(var);
}

KernelStorage::~KernelStorage() {
  // Types are not deleted - they are static.
  for (DFVariableImpl *var: variables) {
    delete var;
  }
}

void KernelStorage::transferFrom(KernelStorage &&storage) {
  // Types are not transferred - they are static.
  variables.merge(std::move(storage.variables));
}

} // namespace dfcxx
