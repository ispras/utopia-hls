//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_PASSES_H
#define DFCIR_PASSES_H

#include "dfcir/DFCIROperations.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include <initializer_list>
#include <memory>

namespace mlir::dfcir {

enum Ops {
  // Empty op. Its latency is always 0.
  UNDEFINED,
  // Arithmetic operations.
  ADD_INT,
  ADD_FLOAT,
  SUB_INT,
  SUB_FLOAT,
  MUL_INT,
  MUL_FLOAT,
  DIV_INT,
  DIV_FLOAT,
  NEG_INT,
  NEG_FLOAT,
  // Bitwise operations.
  AND_INT,
  AND_FLOAT,
  OR_INT,
  OR_FLOAT,
  XOR_INT,
  XOR_FLOAT,
  NOT_INT,
  NOT_FLOAT,
  // Comparison operations.
  LESS_INT,
  LESS_FLOAT,
  LESSEQ_INT,
  LESSEQ_FLOAT,
  GREATER_INT,
  GREATER_FLOAT,
  GREATEREQ_INT,
  GREATEREQ_FLOAT,
  EQ_INT,
  EQ_FLOAT,
  NEQ_INT,
  NEQ_FLOAT,
  // Utility value. Contains the number of elements in the enum.
  COUNT
};

} // namespace mlir::dfcir

struct LatencyConfig {
public:
  std::unordered_map<mlir::dfcir::Ops, uint16_t> internalOps;
  std::unordered_map<std::string, uint16_t> externalOps;

  LatencyConfig() = default;

  LatencyConfig(const LatencyConfig &) = default;

  LatencyConfig(
      std::initializer_list<std::pair<const mlir::dfcir::Ops, uint16_t>> internals,
      std::initializer_list<std::pair<const std::string, uint16_t>> externals
  ) : internalOps(internals), externalOps(externals) {}
};

namespace mlir::dfcir {

using std::unique_ptr;
using mlir::Pass;

unique_ptr<Pass> createDFCIRCombPipelinePassPass(uint64_t stages);

unique_ptr<Pass> createDFCIRToFIRRTLPass();

unique_ptr<Pass> createDFCIRASAPSchedulerPass(LatencyConfig *config);

unique_ptr<Pass> createDFCIRLinearSchedulerPass(LatencyConfig *config);

unique_ptr<Pass> createFIRRTLStubGeneratorPass(llvm::raw_ostream *stream);

} // namespace mlir::dfcir

#include "dfcir/passes/DFCIRPasses.h.inc"

#endif // DFCIR_PASSES_H
