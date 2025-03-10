//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/DFCXX.h"

#include "gtest/gtest.h"

using namespace dfcxx; // For testing purposes only.

#define STUB_NAME "Name"

TEST(DFCXXVarBuilder, BuildStream) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  meta.storage.addType(type);
  DFVariableImpl *var = meta.varBuilder.buildStream(STUB_NAME, DFVariableImpl::IODirection::NONE, &meta, type);
  EXPECT_TRUE(var);
  meta.storage.addVariable(var);
  EXPECT_TRUE(var->isStream());
  EXPECT_TRUE(!var->isScalar());
  EXPECT_TRUE(!var->isConstant());
  EXPECT_EQ(var->getName(), STUB_NAME);
  EXPECT_EQ(var->getDirection(), DFVariableImpl::IODirection::NONE);
  EXPECT_EQ(&var->getMeta(), &meta);
  EXPECT_EQ(var->getType(), type);

  DFStream *casted = (DFStream *) var;
  EXPECT_TRUE(casted);
}

TEST(DFCXXVarBuilder, BuildScalar) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  meta.storage.addType(type);
  DFVariableImpl *var = meta.varBuilder.buildScalar(STUB_NAME, DFVariableImpl::IODirection::INPUT, &meta, type);
  EXPECT_TRUE(var);
  meta.storage.addVariable(var);
  EXPECT_TRUE(!var->isStream());
  EXPECT_TRUE(var->isScalar());
  EXPECT_TRUE(!var->isConstant());
  EXPECT_EQ(var->getName(), STUB_NAME);
  EXPECT_EQ(var->getDirection(), DFVariableImpl::IODirection::INPUT);
  EXPECT_EQ(&var->getMeta(), &meta);
  EXPECT_EQ(var->getType(), type);

  DFScalar *casted = (DFScalar *) var;
  EXPECT_TRUE(casted);
}

TEST(DFCXXVarBuilder, BuildUnsignedConstant) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildFixed(FixedType::SignMode::UNSIGNED, 64, 0);
  meta.storage.addType(type);
  DFConstant::Value val { .uint_ = uint64_t(-1) };
  DFVariableImpl *var = meta.varBuilder.buildConstant(&meta, type, val);
  EXPECT_TRUE(var);
  meta.storage.addVariable(var);
  EXPECT_TRUE(!var->isStream());
  EXPECT_TRUE(!var->isScalar());
  EXPECT_TRUE(var->isConstant());
  EXPECT_EQ(var->getName(), "");
  EXPECT_EQ(var->getDirection(), DFVariableImpl::IODirection::NONE);
  EXPECT_EQ(&var->getMeta(), &meta);
  EXPECT_EQ(var->getType(), type);

  DFConstant *casted = (DFConstant *) var;
  EXPECT_TRUE(casted);
  EXPECT_EQ(casted->getInt(), -1);
  EXPECT_EQ(casted->getUInt(), uint64_t(0 - 1));
  EXPECT_EQ(casted->getKind(), DFConstant::TypeKind::UINT);
}

TEST(DFCXXVarBuilder, BuildSignedConstant) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildFixed(FixedType::SignMode::SIGNED, 63, 0);
  meta.storage.addType(type);
  DFConstant::Value val { .int_ = int64_t(-1) };
  DFVariableImpl *var = meta.varBuilder.buildConstant(&meta, type, val);
  EXPECT_TRUE(var);
  meta.storage.addVariable(var);
  EXPECT_TRUE(!var->isStream());
  EXPECT_TRUE(!var->isScalar());
  EXPECT_TRUE(var->isConstant());
  EXPECT_EQ(var->getName(), "");
  EXPECT_EQ(var->getDirection(), DFVariableImpl::IODirection::NONE);
  EXPECT_EQ(&var->getMeta(), &meta);
  EXPECT_EQ(var->getType(), type);

  DFConstant *casted = (DFConstant *) var;
  EXPECT_TRUE(casted);
  EXPECT_EQ(casted->getInt(), -1);
  EXPECT_EQ(casted->getUInt(), uint64_t(0 - 1));
  EXPECT_EQ(casted->getKind(), DFConstant::TypeKind::INT);
}

TEST(DFCXXVarBuilder, BuildFloatConstant) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildFloat(8, 31);
  meta.storage.addType(type);
  DFConstant::Value val { .double_ = 2.5 };
  DFVariableImpl *var = meta.varBuilder.buildConstant(&meta, type, val);
  EXPECT_TRUE(var);
  meta.storage.addVariable(var);
  EXPECT_TRUE(!var->isStream());
  EXPECT_TRUE(!var->isScalar());
  EXPECT_TRUE(var->isConstant());
  EXPECT_EQ(var->getName(), "");
  EXPECT_EQ(var->getDirection(), DFVariableImpl::IODirection::NONE);
  EXPECT_EQ(&var->getMeta(), &meta);
  EXPECT_EQ(var->getType(), type);

  DFConstant *casted = (DFConstant *) var;
  EXPECT_TRUE(casted);
  EXPECT_EQ(casted->getDouble(), 2.5);
  EXPECT_EQ(casted->getKind(), DFConstant::TypeKind::FLOAT);
}
