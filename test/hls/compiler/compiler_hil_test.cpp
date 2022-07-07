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
#include "hls/mapper/mapper.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/topological_balancer.h"

#include <iostream>
#include <fstream>
#include <memory>

using namespace eda::hls::compiler;
using namespace eda::hls::parser::hil;
using namespace eda::hls::library;
using namespace eda::hls::scheduler;
using namespace eda::hls::mapper;

int compilerHilTest(const std::string &inputLibraryPath,
                    const std::string &relativeCompPath,
                    const std::string &inputFilePath,
                    const std::string &outputFirrtlName,
                    const std::string &outputVerilogLibraryName,
                    const std::string &outputVerilogTopModuleName,
                    const std::string &outputDirName,
                    const std::string &outputTestName) {
  Indicators indicators;
  // Optimization criterion and constraints.
  eda::hls::model::Criteria criteria(
    PERF,
    eda::hls::model::Constraint<unsigned>(40000, 500000),                                // Frequency (kHz)
    eda::hls::model::Constraint<unsigned>(1000,  500000),                                // Performance (=frequency)
    eda::hls::model::Constraint<unsigned>(0,     1000),                                  // Latency (cycles)
    eda::hls::model::Constraint<unsigned>(),                                             // Power (does not matter)
    eda::hls::model::Constraint<unsigned>(1,     10000000));

  std::shared_ptr<Model> model = parse(inputFilePath);

  Library::get().initialize(inputLibraryPath, relativeCompPath);

  Mapper::get().map(*model, Library::get());
  std::map<std::string, Parameters> params =
    ParametersOptimizer<TopologicalBalancer>::get().optimize(criteria,
                                                             *model,
                                                             indicators);

  TopologicalBalancer::get().balance(*model);

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

TEST(CompilerHilTest, CompilerHilTestIdct) {
  EXPECT_EQ(compilerHilTest("./test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "./test/data/hil/idct.hil",
                            "outputIdctFirrtl.mlir",
                            "outputIdctVerilogLibrary.v",
                            "outputIdctVerilogTopModule.v",
                            "./output/test/hil/idct/",
                            "outputIdctTestbench.v"), 0);
}

TEST(CompilerHilTest, CompilerHilTestTest) {
  EXPECT_EQ(compilerHilTest("./test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "./test/data/hil/test.hil",
                            "outputTestFirrtl.mlir",
                            "outputTestVerilogLibrary.v",
                            "outputTestVerilogTopModule.v",
                            "./output/test/hil/test/",
                            "outputTestTestbench.v"), 0);
}

TEST(CompilerHilTest, CompilerHilTestFeedback) {
  EXPECT_EQ(compilerHilTest("./test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "./test/data/hil/feedback.hil",
                            "outputFeedbackFirrtl.mlir",
                            "outputFeedbackVerilogLibrary.v",
                            "outputFeedbackVerilogTopModule.v",
                            "./output/test/hil/feedback",
                            "outputFeedbackTestbench.v"), 0);
}
