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
#include "hls/xmlparser/xmlparser.h"

#include <iostream>
#include <fstream>
#include <memory>

using namespace eda::hls::compiler;
using namespace eda::hls::parser::hil;
using namespace eda::hls::xmlparser;

int compileSimpleHilTest(const std::string &inputFilePath,
                         const std::string &outputFirrtlName,
                         const std::string &outputVerilogName,
                         const std::string &outputDirName) {
  auto compiler = std::make_unique<Compiler>(*parse(inputFilePath));
  auto xmlparser = std::make_unique<XMLParser>("./src/hls/library/IP-XACT/ispras/ip.hw/library_catalog/1.0/library_catalog.1.0.xml");
  xmlparser->parseIPXACT();
  //auto circuit = compiler->constructCircuit("main");
  //compiler->printFiles(outputFirrtlName, outputVerilogName, outputDirName);
  //compiler->printRndVlogTest(outputDirName + "testbench.v", 10);
  return 0;
}

TEST(CompilerTest, CompileTestIdctTest) {
  EXPECT_EQ(compileSimpleHilTest("./test/data/hil/idct.hil",
                                 "outputFirrtlIdct.mlir",
                                 "outputVerilogIdct.v",
                                 "./test/data/hil/idct/"), 0);
}

/*TEST(CompilerTest, CompileTestHilTest) {
  EXPECT_EQ(compileSimpleHilTest("./test/data/hil/test.hil",
                                 "outputFirrtlTest.mlir",
                                 "outputVerilogTest.v",
                                 "./test/data/hil/test/"), 0);
}*/
