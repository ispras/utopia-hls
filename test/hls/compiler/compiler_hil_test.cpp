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
#include <filesystem>
#include <fstream>
#include <memory>

using namespace eda::hls::compiler;
using namespace eda::hls::parser::hil;
using namespace eda::hls::library;
using namespace eda::hls::scheduler;
using namespace eda::hls::mapper;

namespace fs = std::filesystem;

int compilerHilTest(const std::string &inHilSubPath,
                    const std::string &inLibSubPath,
                    const std::string &relCatPath,
                    const std::string &outFirName,
                    const std::string &outVlogLibName,
                    const std::string &outVlogTopName,
                    const std::string &outSubPath,
                    const std::string &outTestName) {
  Indicators indicators;
  // Optimization criterion and constraints.
  eda::hls::model::Criteria criteria(
    PERF,
    eda::hls::model::Constraint<unsigned>(40000, 500000),                                // Frequency (kHz)
    eda::hls::model::Constraint<unsigned>(1000,  500000),                                // Performance (=frequency)
    eda::hls::model::Constraint<unsigned>(0,     1000),                                  // Latency (cycles)
    eda::hls::model::Constraint<unsigned>(),                                             // Power (does not matter)
    eda::hls::model::Constraint<unsigned>(1,     10000000));

  const fs::path fsInHilSubPath = inHilSubPath;
  const std::string inHilPath = std::string(getenv("UTOPIA_HOME"))
      / fsInHilSubPath;

  std::shared_ptr<Model> model = parse(inHilPath);

  const fs::path fsInLibSubPath = inLibSubPath;
  const std::string inLibPath = std::string(getenv("UTOPIA_HOME"))
      / fsInLibSubPath;

  Library::get().initialize(inLibPath, relCatPath);

  Mapper::get().map(*model, Library::get());
  std::map<std::string, Parameters> params =
    ParametersOptimizer<TopologicalBalancer>::get().optimize(criteria,
                                                             *model,
                                                             indicators);

  TopologicalBalancer::get().balance(*model);

  auto compiler = std::make_unique<Compiler>();
  auto circuit = compiler->constructFirrtlCircuit(*model, "main");

  const fs::path fsOutSubPath = outSubPath;

  const std::string outPath = std::string(getenv("UTOPIA_HOME"))
      / fsOutSubPath;
  circuit->printFiles(outFirName,
                      outVlogLibName,
                      outVlogTopName,
                      outPath);

  // generate random test of the specified length in ticks
  const int testLength = 10;
  circuit->printRndVlogTest(*model,
                            outPath,
                            outTestName,
                            testLength);

  Library::get().finalize();
  return 0;
}

TEST(CompilerHilTest, CompilerHilTestIdct) {
  EXPECT_EQ(compilerHilTest("test/data/hil/idct.hil",
                            "test/data/ipx/ispras/ip.hw/",
                            "catalog/1.0/catalog.1.0.xml",
                            "idctFir.mlir",
                            "idctLib.v",
                            "idctTop.v",
                            "output/test/hil/idct/",
                            "idctTestBench.v"), 0);
}

TEST(CompilerHilTest, CompilerHilTestTest) {
  EXPECT_EQ(compilerHilTest("test/data/hil/test.hil",
                            "test/data/ipx/ispras/ip.hw",
                            "catalog/1.0/catalog.1.0.xml",
                            "testFir.mlir",
                            "testLib.v",
                            "testTop.v",
                            "output/test/hil/test/",
                            "testTestBench.v"), 0);
}

TEST(CompilerHilTest, CompilerHilTestFeedback) {
  EXPECT_EQ(compilerHilTest("test/data/hil/feedback.hil",
                            "test/data/ipx/ispras/ip.hw/",
                            "catalog/1.0/catalog.1.0.xml",
                            "feedbackFir.mlir",
                            "feedbackLib.v",
                            "feedbackTop.v",
                            "output/test/hil/feedback/",
                            "feedbackTestBench.v"), 0);
}
