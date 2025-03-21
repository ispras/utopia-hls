//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "config.h"
#include "dfcxx/DFCXX.h"
#include "options.h"

#include "easylogging++.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

INITIALIZE_EASYLOGGINGPP

// User-defined function to specify functional behaviour of top-level kernel.
std::unique_ptr<dfcxx::Kernel> start();

//===-----------------------------------------------------------------------===/
// High-Level Synthesis
//===-----------------------------------------------------------------------===/

struct HlsContext {
  HlsContext(const HlsOptions &options) : options(options) {}

  const HlsOptions &options;
};

//===-----------------------------------------------------------------------===/
// DFCxx Simulation.
//===-----------------------------------------------------------------------===/

struct SimContext {
  SimContext(const SimOptions &options): options(options) {}

  const SimOptions &options;
};

int hlsMain(const HlsContext &context) {
  auto kernel = start();
  if (!kernel->check()) { return 1; }

  dfcxx::DFOptionsConfig optionsCfg = context.options.optionsCfg;

  if (context.options.asapScheduler) {
    optionsCfg.scheduler = dfcxx::Scheduler::ASAP;
  } else if (context.options.lpScheduler) {
    optionsCfg.scheduler = dfcxx::Scheduler::Linear;
  } else {
    optionsCfg.scheduler = dfcxx::Scheduler::CombPipelining;
  }

  return !kernel->compile(context.options.latencyCfg,
                          context.options.outNames,
                          optionsCfg);
}

int simMain(const SimContext &context) {
  auto kernel = start();
  return !kernel->simulate(context.options.inFilePath,
                           context.options.outFilePath);
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
    options.initialize(JSON_CONFIG_PATH, argc, argv);
  }
  catch(const CLI::ParseError &e) {
    return options.exit(e);
  }
  
  if (options.hls) {
    return hlsMain(HlsContext(options.hls));
  } else {
    return simMain(SimContext(options.sim));
  }
}
