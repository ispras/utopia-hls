//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_VAR_BUILDER_H
#define DFCXX_VAR_BUILDER_H

#include "dfcxx/vars/vars.h"

namespace dfcxx {

struct KernMeta; // Forward declaration to omit cyclic dependency.

class VarBuilder {
public:
  DFVariableImpl *buildStream(const std::string &name,
                              DFVariableImpl::IODirection direction,
                              KernMeta *meta, DFTypeImpl *type);

  DFVariableImpl *buildScalar(const std::string &name,
                              DFVariableImpl::IODirection direction,
                              KernMeta *meta, DFTypeImpl *type);

  DFVariableImpl *buildConstant(KernMeta *meta,
                                DFTypeImpl *type,
                                DFConstant::Value value);

  DFVariableImpl *buildClone(DFVariableImpl *var);
};

} // namespace dfcxx

#endif // DFCXX_VAR_BUILDER_H
