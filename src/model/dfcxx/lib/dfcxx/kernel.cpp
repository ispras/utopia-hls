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

bool Kernel::compile(const DFLatencyConfig &config, const Scheduler &sched) {
  DFCIRBuilder builder(config);
  auto compiled = builder.buildModule(this);
  llvm::raw_fd_ostream &out = llvm::outs();
  return DFCIRConverter(config).convertAndPrint(compiled, out, sched);
}

bool Kernel::compile(const DFLatencyConfig &config,
                     const std::string &filePath,
                     const Scheduler &sched) {
  if (filePath.empty()) { return compile(config, sched); }
  DFCIRBuilder builder(config);
  auto compiled = builder.buildModule(this);
  std::error_code ec;
  llvm::raw_fd_ostream out(filePath, ec);
  bool result = DFCIRConverter(config).convertAndPrint(compiled, out, sched);
  out.close();
  return result;
}

} // namespace dfcxx

