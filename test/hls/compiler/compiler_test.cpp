//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "hls/compiler/compiler.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"

#include <iostream>
#include <fstream>
#include <memory>

using namespace eda::hls::compiler;
using namespace eda::hls::parser::hil;

int compileSimpleHilTest(const std::string &inputFileName,
                         const std::string &outputFirrtlName,
                         const std::string &outputVerilogName,
                         const std::string &testName) {
  auto compiler = std::make_unique<Compiler>(*parse(Compiler::relativePath + inputFileName));
  auto circuit = compiler->constructCircuit();
  circuit->printFiles(outputFirrtlName, outputVerilogName, testName);

  return 0;
}

TEST(CompilerTest, CompileTestIdctTest) {
  EXPECT_EQ(compileSimpleHilTest("idct.hil",
                                 "outputFirrtlIdct.mlir",
                                 "outputVerilogIdct.v",
                                 "idct"), 0);
}

TEST(CompilerTest, CompileTestHilTest) {
  EXPECT_EQ(compileSimpleHilTest("test.hil",
                                 "outputFirrtlTest.mlir",
                                 "outputVerilogTest.v",
                                 "test"), 0);
}
