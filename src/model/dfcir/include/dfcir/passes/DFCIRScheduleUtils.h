//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_SCHEDULING_UTILS_H
#define DFCIR_SCHEDULING_UTILS_H

#include "dfcir/passes/DFCIRGraph.h"
#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/passes/DFCIRPassesUtils.h"

#include <unordered_map>
#include <utility>

namespace mlir::dfcir::utils {

struct SchedChannel;

struct SchedNode : public GraphNode<SchedChannel> {
  int32_t latency;

  SchedNode(NodeID id, Operation *op, int32_t latency) : GraphNode(id, op) {
    this->latency = latency;
  }

  SchedNode() = default;

  SchedNode(const SchedNode &node) = default;

  ~SchedNode() = default;
};

struct SchedChannel : public GraphChannel<SchedNode> {
  int32_t offset;

  SchedChannel(SchedNode *source, SchedNode *target, int8_t valInd,
               int32_t offset) : GraphChannel(source, target, valInd) {
    this->offset = offset;
  }

  SchedChannel() = default;

  SchedChannel(const SchedChannel &) = default;

  ~SchedChannel() = default;
};

} // namespace mlir::dfcir::utils

namespace mlir::dfcir::utils {

class SchedGraph : public Graph<SchedNode, SchedChannel> {
public:
  typedef std::unordered_map<SchedNode *, int32_t> Latencies;

  using Graph<SchedNode, SchedChannel>::Graph;

  ~SchedGraph() override = default;

  void applyConfig(const LatencyConfig &cfg);

private:
  std::pair<Value, int32_t> findNearestNodeValue(Value value);
  SchedNode *process(Operation *op, OpNodeMap &map) override;
};

} // namespace mlir::dfcir::utils

#endif // DFCIR_SCHEDULING_UTILS_H
