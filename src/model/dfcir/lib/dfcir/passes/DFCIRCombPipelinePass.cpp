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
#define GEN_PASS_DECL_DFCIRCOMBPIPELINEPASS
#define GEN_PASS_DEF_DFCIRCOMBPIPELINEPASS

#include "dfcir/passes/DFCIRPasses.h.inc"

class DFCIRCombPipelinePass
    : public impl::DFCIRCombPipelinePassBase<DFCIRCombPipelinePass> {

public:
  explicit DFCIRCombPipelinePass(const DFCIRCombPipelinePassOptions &options)
      : impl::DFCIRCombPipelinePassBase<DFCIRCombPipelinePass>(options) {}

  void runOnOperation() override {
    // 1. Graph construction:
    // Needs: DFCIR module
    // Produces: CombGraph
    //   I. Node(operation, latency, layerId, layerIt)
    //      If operation == nullptr, then it is a fake node (for path lengths equalization only).
    //   II. Channel(source, target, valInd)
    //   III. CombGraph:
    //        * std::unordered_set<Node *> nodes;
    //        * std::unordered_set<Node *> startNodes;
    //        * std::unordered_set<Node *> endNodes;
    //        * std::unordered_map<Node *, std::vector<Channel *>> inputs;
    //        * std::unordered_map<Node *, std::vector<Channel *>> outputs;
    //        * std::unordered_map<mlir::detail::ValueImpl *, ConnectOp> connectionMap;

    // 2. Layer partitioning:
    // Needs: CombGraph
    // Produces: Updated CombGraph (with layer specs),
    //           std::vector<std::forward_list<Node *>> layers
    //           std::vector<uint64_t> layerWeights
    //

    // 3. Path equalizaition:
    // Needs: CombGraph,
    //        std::vector<std::forward_list<Node *>> layers
    // Produces: Updated CombGraph (with fake nodes and equal paths)

    // 4. Cascade partitioning:
    // Needs: std::vector<std::forward_list<Node *>> layers,
    //        std::vector<uint64_t> layerWeights
    // Produces: std::vector<uint64_t> cascades; // Every cascade has a corresponding idx of its rightmost layer

    // 5. FIFO insertion:
    // Needs: CombGraph,
    //        std::vector<std::forward_list<Node *>> layers,
    //        std::vector<uint64_t> cascades;
    // Produces: 1-stage FIFO's on cascades boundaries
  }
};

std::unique_ptr<mlir::Pass> createDFCIRCombPipelinePassPass(uint64_t stages) {
  DFCIRCombPipelinePassOptions options;
  options.stages = stages;
  return std::make_unique<DFCIRCombPipelinePass>(options);
}

} // namespace mlir::dfcir
