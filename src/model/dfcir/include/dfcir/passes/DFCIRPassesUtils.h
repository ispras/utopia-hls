//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_PASSES_UTILS_H
#define DFCIR_PASSES_UTILS_H

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/DFCIROperations.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlir::utils {

template <typename OpTy>
inline OpTy findFirstOccurence(Operation *op) {
  Operation *result = nullptr;
  op->template walk<mlir::WalkOrder::PreOrder>(
          [&](Operation *found) -> mlir::WalkResult {
            if (llvm::dyn_cast<OpTy>(found)) {
              result = found;
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          });
  return llvm::dyn_cast<OpTy>(result);
}

} // namespace mlir::utils

namespace mlir::dfcir::utils {

void eraseOffsets(mlir::Operation *op);

Ops resolveInternalOpType(mlir::Operation *op);

std::string opTypeToString(const Ops &opType);

} // namespace mlir::dfcir::utils

#endif // DFCIR_PASSES_UTILS_H
