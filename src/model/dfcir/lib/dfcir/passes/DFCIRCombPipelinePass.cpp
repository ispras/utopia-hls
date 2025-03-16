//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"
#include "dfcir/passes/DFCIRPipeliningUtils.h"
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
  using CombGraph = utils::CombGraph;
  using NodeLayers = utils::CombGraph::NodeLayers;
  using LayerLatencies = utils::CombGraph::LayerLatencies;
  using LayerCascades = utils::CombGraph::LayerCascades;

  void runOnOperation() override {
    // Convert kernel into graph.
    CombGraph graph;
    graph.constructFrom(llvm::cast<ModuleOp>(getOperation()));

    // Assign nodes to layers and calculate each layer's weight.
    NodeLayers nodeLayers;
    LayerLatencies layerLatencies;
    graph.divideIntoLayers(nodeLayers, layerLatencies);

    // Assign layers to cascades.
    LayerCascades layerCascades;
    graph.divideIntoCascades(stages, layerLatencies, layerCascades);

    // Calculate what FIFOs need to be inserted.
    CombGraph::Buffers buffers = graph.calculateFIFOs(nodeLayers, layerCascades);

    // Insert buffers.
    graph.insertBuffers(this->getContext(), buffers);
  }
};

std::unique_ptr<mlir::Pass> createDFCIRCombPipelinePassPass(uint64_t stages) {
  DFCIRCombPipelinePassOptions options;
  options.stages = stages;
  return std::make_unique<DFCIRCombPipelinePass>(options);
}

} // namespace mlir::dfcir
