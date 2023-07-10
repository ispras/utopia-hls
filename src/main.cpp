//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "config.h"
#include "hls/compiler/compiler.h"
#include "hls/mapper/mapper.h"
#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/latency_solver.h"
#include "hls/scheduler/param_optimizer.h"
#include "options.h"
#include "utils/string.h"

#include "easylogging++.h"

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>

INITIALIZE_EASYLOGGINGPP

//===-----------------------------------------------------------------------===/
// High-Level Synthesis
//===-----------------------------------------------------------------------===/

struct HlsContext {
  using Model = eda::hls::model::Model;
  using Indicators = eda::hls::model::Indicators;
  using Parameters = eda::hls::model::Parameters;
  using Criteria = eda::hls::model::Criteria;
  template<typename T>
  using Constraint = eda::hls::model::Constraint<T>;

  using Mapper = eda::hls::mapper::Mapper;
  using Library = eda::hls::library::Library;
  using Method = eda::hls::scheduler::LatencyLpSolver;
  using Optimizer = eda::hls::scheduler::ParametersOptimizer<Method>;
  using Compiler = eda::hls::compiler::Compiler;

  HlsContext(const std::string &file, const HlsOptions &options):
    file(file),
    options(options),
    criteria(
        Indicator::PERF,
        Constraint<unsigned>(40000, 500000), // Frequency (kHz)
        Constraint<unsigned>(1000, 500000),  // Performance (=frequency)
        Constraint<unsigned>(0, 1000),       // Latency (cycles)
        Constraint<unsigned>(),              // Power (does not matter)
        Constraint<unsigned>(1, 10000000))   // Area (number of LUTs)
    {}

  const std::string file;
  const HlsOptions &options;

  Criteria criteria;
  std::shared_ptr<Model> model;
  Indicators indicators;
  std::map<std::string, Parameters> params;
};

bool parse(HlsContext &context) {
  LOG(INFO) << "HLS parse: " << context.file;

  context.model = eda::hls::parser::hil::parse(context.file);

  if (context.model == nullptr) {
    LOG(ERROR) << "Could not parse the file";
    return false;
  }

  std::cout << "------ HLS model #0 ------" << std::endl;
  std::cout << *context.model << std::endl;

  return true;
}

bool optimize(HlsContext &context) {
  context.model->save();

  // Map the model nodes to meta elements.
  HlsContext::Mapper::get().map(*context.model, HlsContext::Library::get());
  // Optimize the model.
  context.params = HlsContext::Optimizer::get().optimize(
    context.criteria, *context.model, context.indicators);

  context.model->save();

  std::cout << "------ HLS model #1 ------" << std::endl;
  std::cout << *context.model << std::endl;

  if (!context.options.outDot.empty()) {
    std::ofstream output(context.options.outDir + "/" + context.options.outDot);
    eda::hls::model::printDot(output, *context.model);
    output.close();
  }

  return true;
}

bool compile(HlsContext &context) {
  auto compiler = std::make_unique<HlsContext::Compiler>();

  auto circuit = compiler->constructFirrtlCircuit(*context.model, "main");
  circuit->printFiles(context.options.outMlir,
                      context.options.outLib,
                      context.options.outTop,
                      context.options.outDir);

  if (!context.options.outTest.empty()) {
    circuit->printRndVlogTest(*context.model,
                              context.options.outDir,
                              context.options.outTest,
                              10);
  }

  return true;
}

int hlsMain(HlsContext &context) {
  if (!parse(context))    { return -1; }
  if (!optimize(context)) { return -1; }
  if (!compile(context))  { return -1; }

  return 0;
}

int main(int argc, char **argv) {
  START_EASYLOGGINGPP(argc, argv);

  std::stringstream title;
  std::stringstream version;

  version << VERSION_MAJOR << "." << VERSION_MINOR;

  title << "Utopia EDA " << version.str() << " | ";
  title << "Copyright (c) " << YEAR_STARTED << "-" << YEAR_CURRENT << " ISPRAS";

  Options options(title.str(), version.str());

  try {
    options.initialize("config.json", argc, argv);

    if (options.hls.files().empty()) {
      throw CLI::CallForAllHelp();
    }
  } catch(const CLI::ParseError &e) {
    return options.exit(e);
  }

  int result = 0;

  for (auto file : options.hls.files()) {
    HlsContext context(file, options.hls);
    result |= hlsMain(context);
  }

  return result;
}
