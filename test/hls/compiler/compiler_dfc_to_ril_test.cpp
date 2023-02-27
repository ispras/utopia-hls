//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/compiler/compiler.h"
#include "hls/library/internal/ril/element_internal_ril.h"
#include "hls/library/library.h"
#include "hls/mapper/mapper.h"
#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/dfc/dfc.h"
#include "hls/parser/dfc/internal/builder.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/topological_balancer.h"
#include "util/string.h"

#include "gtest/gtest.h"

#include <array>
#include <filesystem>
#include <fstream>

using ElementInternalRil = eda::hls::library::internal::ril::ElementInternalRil;

namespace fs = std::filesystem;

DFC_KERNEL(VectorSum) {
  DFC_KERNEL_CTOR(VectorSum) {
    static const int SIZE = 4;

    std::array<dfc::stream<dfc::sint16>, SIZE> lhs;
    std::array<dfc::stream<dfc::sint16>, SIZE> rhs;
    std::array<dfc::stream<dfc::sint16>, SIZE> res;

    for (std::size_t i = 0; i < lhs.size(); i++) {
      res[i] = lhs[i] + rhs[i];
    }
  }
};

DFC_KERNEL(VectorSub) {
  DFC_KERNEL_CTOR(VectorSub) {
    static const int SIZE = 4;

    std::array<dfc::stream<dfc::sint16>, SIZE> lhs;
    std::array<dfc::stream<dfc::sint16>, SIZE> rhs;
    std::array<dfc::stream<dfc::sint16>, SIZE> res;

    for (std::size_t i = 0; i < lhs.size(); i++) {
      res[i] = lhs[i] - rhs[i];
    }
  }
};

DFC_KERNEL(ScalarMul) {
  DFC_KERNEL_CTOR(ScalarMul) {
    static const int SIZE = 3;

    std::array<dfc::stream<dfc::sint16>, SIZE> lhs;
    std::array<dfc::stream<dfc::sint16>, SIZE> rhs;
    dfc::stream<dfc::sint16> res;

    res = lhs[0] * rhs[0]; 
    for (std::size_t i = 1; i < lhs.size(); i++) {
      res += lhs[i] * rhs[i];
    }
  }
};



int compilerDfcToRilTest(const dfc::kernel &kernel,
                         const Signature &nodeTypeSignature,
                         const std::string &outSubPath,
                         const std::string &outTestName) {
  const std::string funcName = kernel.name;

  auto &builder = eda::hls::parser::dfc::Builder::get();
  std::shared_ptr<Model> model = builder.create(funcName, funcName);

  auto *nodeType = model->findNodetype(nodeTypeSignature);

  uassert(nodeType != nullptr, "Nodetype " + nodeTypeSignature.name + 
                               " not found!\n");

  auto hwConfing = HWConfig("", "", "");
  auto metaElement = ElementInternalRil::create(*nodeType,
                                                hwConfing);

  uassert(metaElement != nullptr, "MetaElement " + 
                                  nodeTypeSignature.name + 
                                  " is nullptr!");

  auto element = metaElement->construct();

  fs::path basePath = std::getenv("UTOPIA_HOME");
  fs::path fullPath = basePath / outSubPath / outTestName;

  std::ofstream outputFile;
  outputFile.open(fullPath.string());

  uassert(outputFile, "Could not open the file: " + fullPath.string() + "!\n");

  outputFile << element->ir;

  outputFile.close();

  return 0;
}

TEST(CompilerDfcToRilTest, CompilerDfcToRilTestVectorSum) {
  dfc::params args;
  VectorSum kernel(args);
  Signature signature("ADD_2x1",
                      { "fixed_16_0_1", "fixed_16_0_1" },
                      { "fixed_16_0_1" });

  EXPECT_EQ(compilerDfcToRilTest(kernel,
                                 signature,
                                 "test/data/ril/",
                                 "add.ril"), 0);
}

TEST(CompilerDfcToRilTest, CompilerDfcToRilTestVectorMul) {
  dfc::params args;
  ScalarMul kernel(args);
  Signature signature("MUL_2x1",
                    { "fixed_16_0_1", "fixed_16_0_1" },
                    { "fixed_16_0_1" });
  EXPECT_EQ(compilerDfcToRilTest(kernel,
                                 signature,
                                 "test/data/ril",
                                 "mul.ril"), 0);
}

TEST(CompilerDfcToRilTest, CompilerDfcToRilTestVectorSub) {
  dfc::params args;
  VectorSub kernel(args);
  Signature signature("SUB_2x1",
                    { "fixed_16_0_1", "fixed_16_0_1" },
                    { "fixed_16_0_1" });
  EXPECT_EQ(compilerDfcToRilTest(kernel,
                                 signature,
                                 "test/data/ril",
                                 "sub.ril"), 0);
}