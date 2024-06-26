//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CONVERTER_H
#define DFCXX_CONVERTER_H

#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcxx/typedefs.h"
#include "mlir/IR/BuiltinOps.h"

#include <string>

namespace dfcxx {

    class DFCIRConverter {
    private:
        LatencyConfig config;
    public:
        explicit DFCIRConverter(const DFLatencyConfig &config);
        bool convertAndPrint(mlir::ModuleOp module, llvm::raw_fd_ostream &out, const Scheduler &sched);
    };

} // namespace dfcxx

#endif // DFCXX_CONVERTER_H
