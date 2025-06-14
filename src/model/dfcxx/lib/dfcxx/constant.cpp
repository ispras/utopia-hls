//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/constant.h"
#include "dfcxx/vars/constant.h"

namespace dfcxx {

Constant::Constant(KernelMeta &meta) : meta(meta) {}

DFVariable Constant::var(const DFType &type, int64_t value) {
  DFConstant::Value constVal { .int_ = value };
  return DFConstant::createOrUseConst(type, constVal);
}

DFVariable Constant::var(const DFType &type, uint64_t value) {
  DFConstant::Value constVal { .uint_ = value };
  return DFConstant::createOrUseConst(type, constVal);
}

DFVariable Constant::var(const DFType &type, double value) {
  DFConstant::Value constVal { .double_ = value };
  return DFConstant::createOrUseConst(type, constVal);
}

} // namespace dfcxx
