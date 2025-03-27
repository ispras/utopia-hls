//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

#define STUB_NAME "Name"

TEST(DFCXXVarsConstant, AddSameConstant) {
  using namespace dfcxx;

  KernelMeta meta;
  DFConstant::Value val { .uint_ = 2 };

  DFTypeImpl *type =
      meta.typeBuilder.buildFixed(FixedType::SignMode::UNSIGNED, 32, 0);
  meta.storage.addType(type);
  DFVariableImpl *const1 = DFConstant::createOrUseConst(&meta, type, val);

  DFVariableImpl *const2 = DFConstant::createOrUseConst(&meta, type, val);

  EXPECT_EQ(const1, const2);
}

TEST(DFCXXVarsConstant, AddSameConstantDifferentWidth) {
  using namespace dfcxx;

  KernelMeta meta;
  DFConstant::Value val { .uint_ = 2 };

  DFTypeImpl *type1 =
      meta.typeBuilder.buildFixed(FixedType::SignMode::UNSIGNED, 32, 0);
  meta.storage.addType(type1);
  DFVariableImpl *const1 = DFConstant::createOrUseConst(&meta, type1, val);

  DFTypeImpl *type2 =
      meta.typeBuilder.buildFixed(FixedType::SignMode::UNSIGNED, 20, 0);
  meta.storage.addType(type2);
  DFVariableImpl *const2 = DFConstant::createOrUseConst(&meta, type2, val);

  EXPECT_NE(const1, const2);
}
