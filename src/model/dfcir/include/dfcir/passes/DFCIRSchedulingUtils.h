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

struct SchedNode {
  Operation *op;
  int32_t latency;

  explicit SchedNode(Operation *op, int32_t latency = -1) {
    this->op = op;
    this->latency = latency;
  }

  SchedNode() = default;

  SchedNode(const SchedNode &node) = default;

  ~SchedNode() = default;

  bool operator==(const SchedNode &node) const {
    return this->op == node.op;
  }
};

struct SchedChannel {
  SchedNode *source;
  SchedNode *target;
  int8_t valInd;
  int32_t offset;

  SchedChannel(SchedNode *source, SchedNode *target,
               int8_t valInd, int32_t offset = 0) {
    this->source = source;
    this->target = target;
    this->valInd = valInd;
    this->offset = offset;
  }

  SchedChannel() = default;

  SchedChannel(const SchedChannel &) = default;

  ~SchedChannel() = default;

  bool operator==(const SchedChannel &channel) const {
    return this->source == channel.source &&
           this->target == channel.target &&
           this->valInd == channel.valInd;
  }
};

} // namespace mlir::dfcir::utils

template <>
struct std::hash<mlir::dfcir::utils::SchedNode> {
  using SchedNode = mlir::dfcir::utils::SchedNode;

  size_t operator()(const SchedNode &node) const noexcept {
    return std::hash<mlir::Operation *>()(node.op);
  }
};

template <>
struct std::hash<mlir::dfcir::utils::SchedChannel> {
  using SchedNode = mlir::dfcir::utils::SchedNode;
  using SchedChannel = mlir::dfcir::utils::SchedChannel;

  size_t operator()(const SchedChannel &channel) const noexcept {
    return std::hash<SchedNode *>()(channel.target) + 13 + channel.valInd;
  }
};

namespace mlir::dfcir::utils {

class SchedGraph : public Graph<SchedNode, SchedChannel> {
public:
  typedef std::unordered_map<SchedNode *, int32_t> Latencies;
  typedef std::unordered_map<SchedChannel *, int32_t> Buffers;

  using Graph<SchedNode, SchedChannel>::Graph;

  ~SchedGraph() override = default;

  void applyConfig(const LatencyConfig &cfg);

private:
  std::pair<Value, int32_t> findNearestNodeValue(Value value);
  SchedNode *process(Operation *op) override;
};

int32_t calculateOverallLatency(const SchedGraph &graph,
                                SchedGraph::Buffers &buffers,
                                SchedGraph::Latencies *map = {});

} // namespace mlir::dfcir::utils

#endif // DFCIR_SCHEDULING_UTILS_H
