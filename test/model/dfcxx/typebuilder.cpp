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

TEST(DFCXXTypeBuilder, BuildBool) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildBool();
  EXPECT_TRUE(type);
  meta.storage.addType(type);
  EXPECT_TRUE(type->isFixed());
  EXPECT_TRUE(!type->isFloat());
  EXPECT_TRUE(!type->isRawBits());
  EXPECT_EQ(type->getTotalBits(), 1);

  FixedType *casted = (FixedType *) type;
  EXPECT_TRUE(casted);
  EXPECT_TRUE(casted->isBool());
  EXPECT_TRUE(casted->isUnsigned());
  EXPECT_TRUE(!casted->isSigned());
  EXPECT_TRUE(casted->isInt());
  EXPECT_TRUE(casted->isBool());
  EXPECT_EQ(casted->getIntBits(), 1);
  EXPECT_EQ(casted->getFracBits(), 0);
}

TEST(DFCXXTypeBuilder, BuildUnsignedInt) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildFixed(FixedType::SignMode::UNSIGNED, 32, 0);
  EXPECT_TRUE(type);
  meta.storage.addType(type);
  EXPECT_TRUE(type->isFixed());
  EXPECT_TRUE(!type->isFloat());
  EXPECT_TRUE(!type->isRawBits());
  EXPECT_EQ(type->getTotalBits(), 32);

  FixedType *casted = (FixedType *) type;
  EXPECT_TRUE(casted);
  EXPECT_TRUE(!casted->isBool());
  EXPECT_TRUE(casted->isUnsigned());
  EXPECT_TRUE(!casted->isSigned());
  EXPECT_TRUE(casted->isInt());
  EXPECT_TRUE(!casted->isBool());
  EXPECT_EQ(casted->getIntBits(), 32);
  EXPECT_EQ(casted->getFracBits(), 0);
}

TEST(DFCXXTypeBuilder, BuildSignedInt) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildFixed(FixedType::SignMode::SIGNED, 31, 0);
  EXPECT_TRUE(type);
  meta.storage.addType(type);
  EXPECT_TRUE(type->isFixed());
  EXPECT_TRUE(!type->isFloat());
  EXPECT_TRUE(!type->isRawBits());
  EXPECT_EQ(type->getTotalBits(), 32);

  FixedType *casted = (FixedType *) type;
  EXPECT_TRUE(casted);
  EXPECT_TRUE(!casted->isBool());
  EXPECT_TRUE(!casted->isUnsigned());
  EXPECT_TRUE(casted->isSigned());
  EXPECT_TRUE(casted->isInt());
  EXPECT_TRUE(!casted->isBool());
  EXPECT_EQ(casted->getIntBits(), 31);
  EXPECT_EQ(casted->getFracBits(), 0);
}

TEST(DFCXXTypeBuilder, BuildFixed) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildFixed(FixedType::SignMode::SIGNED, 31, 32);
  EXPECT_TRUE(type);
  meta.storage.addType(type);
  EXPECT_TRUE(type->isFixed());
  EXPECT_TRUE(!type->isFloat());
  EXPECT_TRUE(!type->isRawBits());
  EXPECT_EQ(type->getTotalBits(), 64);

  FixedType *casted = (FixedType *) type;
  EXPECT_TRUE(casted);
  EXPECT_TRUE(!casted->isBool());
  EXPECT_TRUE(!casted->isUnsigned());
  EXPECT_TRUE(casted->isSigned());
  EXPECT_TRUE(!casted->isInt());
  EXPECT_TRUE(!casted->isBool());
  EXPECT_EQ(casted->getIntBits(), 31);
  EXPECT_EQ(casted->getFracBits(), 32);
}

TEST(DFCXXTypeBuilder, BuildFloat) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildFloat(8, 23);
  EXPECT_TRUE(type);
  meta.storage.addType(type);
  EXPECT_TRUE(!type->isFixed());
  EXPECT_TRUE(type->isFloat());
  EXPECT_TRUE(!type->isRawBits());
  EXPECT_EQ(type->getTotalBits(), 32);

  FloatType *casted = (FloatType *) type;
  EXPECT_TRUE(casted);
  EXPECT_EQ(casted->getExpBits(), 8);
  EXPECT_EQ(casted->getFracBits(), 23);
}

TEST(DFCXXTypeBuilder, BuildRawBits) {
  KernMeta meta;
  DFTypeImpl *type = meta.typeBuilder.buildRawBits(20);
  EXPECT_TRUE(type);
  meta.storage.addType(type);
  EXPECT_TRUE(!type->isFixed());
  EXPECT_TRUE(!type->isFloat());
  EXPECT_TRUE(type->isRawBits());
  EXPECT_EQ(type->getTotalBits(), 20);

  RawBitsType *casted = (RawBitsType *) type;
  EXPECT_TRUE(casted);
}
