//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernstorage.h"

#include <algorithm>

namespace dfcxx {

dfcxx::DFTypeImpl *KernStorage::addType(dfcxx::DFTypeImpl *type) {
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

DFVariableImpl *KernStorage::addVariable(DFVariableImpl *var,  VarComp cmp) {
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

void KernStorage::deleteVariable(DFVariableImpl *var) {
  variables.erase(var);
}

KernStorage::~KernStorage() {
  for (DFTypeImpl *type: types) {
    delete type;
  }
  for (DFVariableImpl *var: variables) {
    delete var;
  }
}

void KernStorage::transferFrom(KernStorage &&storage) {
  types.merge(std::move(storage.types));
  variables.merge(std::move(storage.variables));
}

} // namespace dfcxx
