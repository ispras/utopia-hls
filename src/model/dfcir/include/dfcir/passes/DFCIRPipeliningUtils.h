//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_PIPELINING_UTILS_H
#define DFCIR_PIPELINING_UTILS_H

#include "dfcir/passes/DFCIRGraph.h"
#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"

#include <unordered_map>

namespace mlir::dfcir::utils {

typedef float CombLatency;
typedef uint64_t CombLayerID;
typedef uint64_t CombCascadeID;

struct CombChannel;

struct CombNode : public GraphNode<CombChannel> {
  CombLatency latency;

  CombNode(NodeID id, Operation *op, CombLatency latency) : GraphNode(id, op) {
    this->latency = latency;
  }

  CombNode() = default;

  CombNode(const CombNode &node) = default;

  ~CombNode() = default;
};

struct CombChannel : public GraphChannel<CombNode> {
  int32_t offset;

  // Same constructor as GraphChannel.
  using GraphChannel<CombNode>::GraphChannel;

  CombChannel() = default;

  CombChannel(const CombChannel &) = default;

  ~CombChannel() = default;
};

} // namespace mlir::dfcir::utils

namespace mlir::dfcir::utils {

class CombGraph : public Graph<CombNode, CombChannel> {
public:
  // The layer index for each node.
  typedef std::vector<CombLayerID> NodeLayers;
  // The weight for each layer.
  typedef std::vector<CombLatency> LayerLatencies;
  // The cascade index for each layer.
  typedef std::vector<CombCascadeID> LayerCascades;

  static constexpr CombLatency floatEps = 1e-6;

  using Graph<CombNode, CombChannel>::Graph;

  ~CombGraph() override = default;

  void divideIntoLayers(NodeLayers &nodeLayers, LayerLatencies &layerLatencies);

  void divideIntoCascades(const uint64_t cascadesCount,
                          const LayerLatencies &layerLatencies,
                          LayerCascades &layerCascades);

  Buffers calculateFIFOs(const uint64_t cascadesCount,
                         const NodeLayers &nodeLayers,
                         const LayerCascades &layersCascades);

private:
  Value findNearestNodeValue(Value value);
  CombNode *process(Operation *op, OpNodeMap &map) override;
};

} // namespace mlir::dfcir::utils

#endif // DFCIR_PIPELINING_UTILS_H
