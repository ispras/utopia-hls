//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <memory>

#include "gtest/gtest.h"

#include "hls/compiler/compiler.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::compiler;
using namespace eda::hls::parser::hil;

int compileSimpleHilTest(const std::string &inputFileName,
                         const std::string &outputFirrtlName,
                         const std::string &outputVerilogName) {
  auto compiler = std::make_unique<Compiler>(*parse(inputFileName));
  auto circuit = compiler->constructCircuit();
  (*circuit).printFiles(outputFirrtlName, outputVerilogName);
  return 0;
}

TEST(CompilerTest, CompileTestIdctTest) {
  EXPECT_EQ(compileSimpleHilTest("test/data/hil/idct.hil",
                                 "test/data/hil/outputFirrtlIdct.mlir",
                                 "test/data/hil/outputVerilogIdct.v"), 0);
}

TEST(CompilerTest, CompileTestHilTest) {
  EXPECT_EQ(compileSimpleHilTest("test/data/hil/test.hil",
                                 "test/data/hil/outputFirrtlTest.mlir",
                                 "test/data/hil/outputVerilogTest.v"), 0);
}
