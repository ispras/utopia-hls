//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CONVERTER_H
#define DFCXX_CONVERTER_H

#include "dfcxx/typedefs.h"

#include "dfcir/conversions/DFCIRPasses.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>

typedef std::vector<llvm::raw_fd_ostream *> OutputStreams;

namespace dfcxx {

class DFCIRConverter {
private:
  LatencyConfig config;
public:
  explicit DFCIRConverter(const DFLatencyConfig &config);
  bool convertAndPrint(mlir::ModuleOp module,
                       OutputStreams &outputStreams,
                       const Scheduler &sched);
};

} // namespace dfcxx

#endif // DFCXX_CONVERTER_H
