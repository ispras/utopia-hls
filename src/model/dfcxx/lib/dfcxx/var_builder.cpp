//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/var_builder.h"

namespace dfcxx {

DFVariableImpl *VarBuilder::buildStream(const std::string &name,
                                        DFVariableImpl::IODirection direction,
                                        KernelMeta *meta,
                                        DFTypeImpl *type) {
  return new DFStream(name, direction, meta, type);
}

DFVariableImpl *VarBuilder::buildScalar(const std::string &name,
                                        DFVariableImpl::IODirection direction,
                                        KernelMeta *meta,
                                        DFTypeImpl *type) {
  return new DFScalar(name, direction, meta, type);
}

DFVariableImpl *VarBuilder::buildConstant(KernelMeta *meta,
                                          DFTypeImpl *type,
                                          DFConstant::Value value) {
  return new DFConstant(meta, type, value);
}

DFVariableImpl *VarBuilder::buildClone(DFVariableImpl *var) {
  return var->clone();
}

} // namespace dfcxx
