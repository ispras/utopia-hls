//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/varbuilders/builder.h"

namespace dfcxx {

DFVariableImpl *VarBuilder::buildStream(const std::string &name,
                                        IODirection direction,
                                        KernMeta &meta,
                                        DFTypeImpl *type) {
  return new DFStream(name, direction, meta, type);
}

DFVariableImpl *VarBuilder::buildScalar(const std::string &name,
                                        IODirection direction,
                                        KernMeta &meta,
                                        DFTypeImpl *type) {
  return new DFScalar(name, direction, helper, type);
}

DFVariableImpl *VarBuilder::buildConstant(KernMeta &meta,
                                          DFTypeImpl *type,
                                          ConstantTypeKind kind,
                                          ConstantValue value) {
  return new DFConstant(helper, type, kind, value);
}

DFVariableImpl *VarBuilder::buildMuxCopy(DFVariableImpl *var, KernMeta &meta) {
  if (var->isConstant()) {
    return buildConstant(meta, var->getType(),
                         ((DFConstant *) var)->getKind(),
                         ConstantValue{});
  } else if (var->isScalar()) {
    return buildScalar("", IODirection::NONE, meta,
                       var->getType());
  } else /* if (var.isStream()) */ {
    return buildStream("", IODirection::NONE, meta,
                       var->getType());
  }
}

} // namespace dfcxx
