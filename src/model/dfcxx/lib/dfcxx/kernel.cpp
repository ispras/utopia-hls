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

#include <iostream>

namespace dfcxx {

Kernel::Kernel() : storage(), typeBuilder(), varBuilder(),
                   graph(), io(graph, typeBuilder, varBuilder, storage),
                   offset(graph, typeBuilder, varBuilder, storage),
                   constant(graph, typeBuilder, varBuilder, storage),
                   control(graph, typeBuilder, varBuilder, storage) {}

DFType Kernel::dfUInt(uint8_t bits) {
  DFTypeImpl *type = typeBuilder.buildFixed(SignMode::UNSIGNED, bits, 0);
  return DFType(storage.addType(type));
}

DFType Kernel::dfInt(uint8_t bits) {
  DFTypeImpl *type = typeBuilder.buildFixed(SignMode::SIGNED, bits, 0);
  return DFType(storage.addType(type));
}

DFType Kernel::dfFloat(uint8_t expBits, uint8_t fracBits) {
  DFTypeImpl *type = typeBuilder.buildFloat(expBits, fracBits);
  return DFType(storage.addType(type));
}

DFType Kernel::dfBool() {
  DFTypeImpl *type = typeBuilder.buildBool();
  return DFType(storage.addType(type));
}


bool Kernel::compile(const DFLatencyConfig &config,
                     const std::vector<std::string> &outputPaths,
                     const Scheduler &sched) {
  DFCIRBuilder builder(config);
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

} // namespace dfcxx
