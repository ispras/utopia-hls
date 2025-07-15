//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

namespace mlir::dfcir {
#define GEN_PASS_DECL_DFCIRCOMBINIFYPASS
#define GEN_PASS_DEF_DFCIRCOMBINIFYPASS

#include "dfcir/passes/DFCIRPasses.h.inc"

class DFCIRCombinifyPass
    : public impl::DFCIRCombinifyPassBase<DFCIRCombinifyPass> {

public:
  explicit DFCIRCombinifyPass()
      : impl::DFCIRCombinifyPassBase<DFCIRCombinifyPass>() {}

  void runOnOperation() override {
    KernelOp kernel = mlir::cast<KernelOp>(getOperation());
    Block &block = kernel.getBody().front();

    for (Scheduled sched: block.getOps<Scheduled>()) {
      sched.setLatency(0);
    }
  }
};

std::unique_ptr<mlir::Pass> createDFCIRCombinifyPass() {
  return std::make_unique<DFCIRCombinifyPass>();
}

} // namespace mlir::dfcir
