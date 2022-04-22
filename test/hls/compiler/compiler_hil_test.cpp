//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "hls/compiler/compiler.h"
#include "hls/library/ipxact_parser.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/dijkstra.h"

#include <iostream>
#include <fstream>
#include <memory>

using namespace eda::hls::compiler;
using namespace eda::hls::parser::hil;
using namespace eda::hls::library;
using namespace eda::hls::scheduler;

int compilerHilTest(const std::string &inputLibraryPath,
                            const std::string &inputFilePath,
                            const std::string &outputFirrtlName,
                            const std::string &outputVerilogName,
                            const std::string &outputDirName) {
  std::shared_ptr<Model> model = parse(inputFilePath);
  IPXACTParser::get().parseCatalog(inputLibraryPath);
  DijkstraBalancer::get().balance(*model);
  auto compiler = std::make_unique<Compiler>(*model);
  auto circuit = compiler->constructCircuit("main");
  compiler->printFiles(outputFirrtlName, outputVerilogName, outputDirName);
  compiler->printRndVlogTest(outputDirName + "testbench.v", 10);
  return 0;
}

TEST(CompilerTest, CompilerTestIdctTest) {
  EXPECT_EQ(compilerHilTest("./test/data/ipx/ispras/ip.hw/catalog/1.0/catalog.1.0.xml",
                                              "./test/data/hil/idct.hil",
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
