//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/dfc/dfc.h"
#include "hls/parser/dfc/internal/builder.h"

#include "gtest/gtest.h"

#include <array>
#include <fstream>
#include <iostream>

DFC_KERNEL(SumFix64) {

  DFC_KERNEL_CTOR(SumFix64) {

    dfc::stream<dfc::uint64> first;
    dfc::stream<dfc::uint64> second;
    dfc::stream<dfc::uint64> sum;

    sum = first + second;
  }
};

DFC_KERNEL(SumReal16) {

  DFC_KERNEL_CTOR(SumReal16) {

    dfc::stream<dfc::float16> first;
    dfc::stream<dfc::float16> second;
    dfc::stream<dfc::float16> sum;

    sum = first + second;
  }
};

DFC_KERNEL(MulComplexReal32) {

  DFC_KERNEL_CTOR(MulComplexReal32) {

    dfc::stream<dfc::complex<dfc::float32>> first;
    dfc::stream<dfc::complex<dfc::float32>> second;
    dfc::stream<dfc::complex<dfc::float32>> sum;

    sum = first * second;
  }
};

DFC_KERNEL(SumTensor2x2Fix32) {

  DFC_KERNEL_CTOR(SumTensor2x2Fix32) {

    dfc::stream<dfc::tensor<dfc::uint32, 2, 2>> first;
    dfc::stream<dfc::tensor<dfc::uint32, 2, 2>> second;
    dfc::stream<dfc::tensor<dfc::uint32, 2, 2>> sum;

    sum = first + second;
  }
};

DFC_KERNEL(SumBits8) {

  DFC_KERNEL_CTOR(SumBits8) {

    dfc::stream<dfc::bits<8>> first;
    dfc::stream<dfc::bits<8>> second;
    dfc::stream<dfc::bits<8>> sum;

    sum = first + second;
  }
};

void dfcModelTypesTest(const dfc::kernel &kernel) {
  const auto funcName = kernel.name;
  auto &builder = eda::hls::parser::dfc::Builder::get();
  std::shared_ptr<Model> model = builder.create(funcName, funcName);
  std::cout << *model << std::endl;

  std::ofstream output("dfc_" + toLower(funcName) + "_test.dot");
  eda::hls::model::printDot(output, *model);
  output.close();
}

TEST(DfcTest, DfcModelTypesSumFix64Test) {
  dfc::params args;
  SumFix64 kernel(args);
  
  dfcModelTypesTest(kernel);
}

TEST(DfcTest, DfcModelTypesSumReal16Test) {
  dfc::params args;
  SumReal16 kernel(args);
  
  dfcModelTypesTest(kernel);
}

TEST(DfcTest, DfcModelTypesMulComplexReal32Test) {
  dfc::params args;
  MulComplexReal32 kernel(args);
  
  dfcModelTypesTest(kernel);
}

TEST(DfcTest, DfcModelTypesSumTensor2x2Fix32Test) {
  dfc::params args;
  SumTensor2x2Fix32 kernel(args);
  
  dfcModelTypesTest(kernel);
}

TEST(DfcTest, DfcModelTypesSumBits8Test) {
  dfc::params args;
  SumBits8 kernel(args);
  
  dfcModelTypesTest(kernel);
}