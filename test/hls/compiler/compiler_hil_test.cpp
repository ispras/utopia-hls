//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "hls/compiler/compiler.h"
#include "hls/library/library.h"
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
                    const std::string &relativeCompPath,
                    const std::string &inputFilePath,
                    const std::string &outputFirrtlName,
                    const std::string &outputVerilogLibraryName,
                    const std::string &outputVerilogTopModuleName,
                    const std::string &outputDirName,
                    const std::string &outputTestName) {

  std::shared_ptr<Model> model = parse(inputFilePath);

  Library::get().initialize(inputLibraryPath, relativeCompPath);

  DijkstraBalancer::get().balance(*model);

  auto compiler = std::make_unique<Compiler>();
  auto circuit = compiler->constructCircuit(*model, "main");
  circuit->printFiles(outputFirrtlName,
                      outputVerilogLibraryName,
                      outputVerilogTopModuleName,
                      outputDirName);

  // generate random test of the specified length in ticks
  const int testLength = 10;
  circuit->printRndVlogTest(*model,
                            outputDirName,
                            outputTestName,
                            model->ind.ticks,
                            testLength);

  Library::get().finalize();
  return 0;
}

TEST(CompilerHilTest, CompilerIdctHilTest) {
  EXPECT_EQ(compilerHilTest("./test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "./test/data/hil/idct.hil",
                            "outputFirrtlIdct.mlir",
                            "outputVerilogLibraryIdct.v",
                            "outputVerilogTopModuleIdct.v",
                            "./output/test/hil/idct",
                            "testbench.v"), 0);
}

TEST(CompilerHilTest, CompilerTestHilTest) {
  EXPECT_EQ(compilerHilTest("./test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "./test/data/hil/test.hil",
                            "outputFirrtlTest.mlir",
                            "outputVerilogLibraryTest.v",
                            "outputVerilogTopModuleTest.v",
                            "./output/test/hil/test",
                            "testbench.v"), 0);
}

TEST(CompilerHilTest, CompilerFeedbackHilTest) {
  EXPECT_EQ(compilerHilTest("./test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "./test/data/hil/feedback.hil",
                            "outputFirrtlFeedback.mlir",
                            "outputVerilogLibraryFeedback.v",
                            "outputVerilogTopModuleFeedback.v",
                            "./output/test/hil/feedback/",
                            "testbench.v"), 0);
}
