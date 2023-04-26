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

namespace fs = std::filesystem;
namespace hilParser = eda::hls::parser::hil;

using Compiler = eda::hls::compiler::Compiler;
using Library = eda::hls::library::Library;
using Mapper = eda::hls::mapper::Mapper;
template<typename T>
using ParametersOptimizer = eda::hls::scheduler::ParametersOptimizer<T>;
using TopologicalBalancer = eda::hls::scheduler::TopologicalBalancer;

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
    Indicator::PERF,
    // Frequency (kHz)
    eda::hls::model::Constraint<unsigned>(40000, 500000),
    // Performance (=frequency)
    eda::hls::model::Constraint<unsigned>(1000,  500000),
    // Latency (cycles)
    eda::hls::model::Constraint<unsigned>(0,     1000),
    // Power (does not matter)
    eda::hls::model::Constraint<unsigned>(),
    // Area (number of LUTs)
    eda::hls::model::Constraint<unsigned>(1,     10000000));

  const fs::path homePath = std::string(getenv("UTOPIA_HLS_HOME"));

  const std::string inHilPath = homePath / inHilSubPath;

  std::shared_ptr<Model> model = hilParser::parse(inHilPath);

  const std::string inLibPath = homePath / inLibSubPath;

  Library::get().initialize();
  Library::get().importLibrary(inLibPath, relCatPath);

  Library::get().excludeLibrary("ip.hw");
  Library::get().includeLibrary("ip.hw");
  Library::get().excludeElementFromLibrary("add", "ip.hw");
  Library::get().includeElementFromLibrary("add", "ip.hw");

  Mapper::get().map(*model, Library::get());
  std::map<std::string, Parameters> params =
      ParametersOptimizer<TopologicalBalancer>::get().optimize(criteria,
                                                               *model,
                                                               indicators);

  TopologicalBalancer::get().balance(*model);

  auto compiler = std::make_unique<Compiler>();
  auto circuit = compiler->constructFirrtlCircuit(*model, "main");

  const fs::path fsOutPath = homePath / outSubPath;
  const std::string outPath = fsOutPath;

  circuit->printFiles(outFirName,
                      outVlogLibName,
                      outVlogTopName,
                      outPath);

  // Generate random test of the specified length in ticks
  const int testLength = 10;
  circuit->printRndVlogTest(*model,
                            outPath,
                            outTestName,
                            testLength);

  std::string pathToOutVlogFiles = fsOutPath / "*.v";

  bool isCompiled = system(("iverilog "
                            + pathToOutVlogFiles).c_str());

  Library::get().finalize();

  return isCompiled;
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
                            "output/test/hil/test",
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
