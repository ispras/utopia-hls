//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/varbuilder.h"

namespace dfcxx {

DFVariableImpl *VarBuilder::buildStream(const std::string &name,
                                        DFVariableImpl::IODirection direction,
                                        KernMeta *meta,
                                        DFTypeImpl *type) {
  return new DFStream(name, direction, meta, type);
}

DFVariableImpl *VarBuilder::buildScalar(const std::string &name,
                                        DFVariableImpl::IODirection direction,
                                        KernMeta *meta,
                                        DFTypeImpl *type) {
  return new DFScalar(name, direction, meta, type);
}

DFVariableImpl *VarBuilder::buildConstant(KernMeta *meta,
                                          DFTypeImpl *type,
                                          DFConstant::Value value) {
  return new DFConstant(meta, type, value);
}

DFVariableImpl *VarBuilder::buildClone(DFVariableImpl *var) {
  return var->clone();
}

} // namespace dfcxx
