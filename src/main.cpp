//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "config.h"
#include "dfcxx/DFCXX.h"
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
  HlsContext(const HlsOptions &options):
    options(options)
    {}

  const HlsOptions &options;
};

// User-defined function to specify functional behaviour of top-level kernel.
//void start(dfcxx::Kernel *kernel);
std::unique_ptr<dfcxx::Kernel> start();

int hlsMain(HlsContext &context) {
  auto kernel = start();
  bool useDijkstra = context.options.dijkstraScheduler;
  kernel->compile(context.options.latConfig,
                  context.options.outFile,
                  (useDijkstra) ? dfcxx::Scheduler::Dijkstra 
                                : dfcxx::Scheduler::Linear);
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
    options.initialize(JSON_CONFIG_PATH, argc, argv);
  }
  catch(const CLI::ParseError &e) {
    return options.exit(e);
  }

  HlsContext context(options.hls);
  return hlsMain(context);
}
