//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/converter.h"
#include "dfcxx/IRbuilders/builder.h"
#include "dfcxx/kernel.h"
#include "dfcxx/simulator.h"
#include "dfcxx/utils.h"

#include <fstream>
#include <iostream>

namespace dfcxx {

Kernel::Kernel() : meta(), io(meta), offset(meta),
                   constant(meta), control(meta) {}

DFType Kernel::dfUInt(uint8_t bits) {
  auto *type = meta.typeBuilder.buildFixed(FixedType::SignMode::UNSIGNED,
                                           bits,
                                           0);
  return meta.storage.addType(type);
}

DFType Kernel::dfInt(uint8_t bits) {
  auto *type = meta.typeBuilder.buildFixed(FixedType::SignMode::SIGNED,
                                           bits,
                                           0);
  return meta.storage.addType(type);
}

DFType Kernel::dfFloat(uint8_t expBits, uint8_t fracBits) {
  auto *type = meta.typeBuilder.buildFloat(expBits, fracBits);
  return meta.storage.addType(type);
}

DFType Kernel::dfBool() {
  auto *type = meta.typeBuilder.buildBool();
  return meta.storage.addType(type);
}

const Graph& Kernel::getGraph() const {
  return meta.graph;
}

bool Kernel::compile(const DFLatencyConfig &config,
                     const std::vector<std::string> &outputPaths,
                     const Scheduler &sched) {
  DFCIRBuilder builder;
  auto compiled = builder.buildModule(this);
  size_t count = outputPaths.size();
  std::vector<llvm::raw_fd_ostream *> outputStreams(count);
  // Output paths strings are converted into output streams.
  for (unsigned i = 0; i < count; ++i) {
    std::error_code ec;
    outputStreams[i] = (!outputPaths[i].empty())
        ? new llvm::raw_fd_ostream(outputPaths[i], ec)
        : nullptr;
  }
  bool result = DFCIRConverter(config).convertAndPrint(compiled,
                                                       outputStreams,
                                                       sched);
  // Every created output stream has to be closed explicitly.
  for (llvm::raw_fd_ostream *stream : outputStreams) {
    if (stream) {
      stream->close();
      delete stream;
    }
  }
  return result;
}

bool Kernel::compile(const DFLatencyConfig &config,
                     const DFOutputPaths &outputPaths,
                     const Scheduler &sched) {
  std::vector<std::string> outPathsStrings(OUT_FORMAT_ID_INT(COUNT), "");
  for (const auto &[id, path] : outputPaths) {
    outPathsStrings[static_cast<uint8_t>(id)] = path;
  }
  return compile(config, outPathsStrings, sched);
}


bool Kernel::simulate(const std::string &inDataPath,
                      const std::string &outFilePath) {
  std::vector<Node> sorted = topSort(meta.graph);
  DFCXXSimulator sim(sorted, meta.graph.getInputs());
  std::ifstream input(inDataPath, std::ios::in);
  if (!input || input.bad() || input.eof() || input.fail() || !input.is_open()) {
    return false;
  }
  std::ofstream output(outFilePath, std::ios::out);
  return sim.simulate(input, output);
}

} // namespace dfcxx
