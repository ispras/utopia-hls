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

namespace mlir::dfcir::utils {
struct Node;
struct Channel;
class Graph;
} // namespace mlir::dfcir::utils

typedef std::unordered_map<mlir::dfcir::utils::Node *, int32_t> Latencies;
typedef std::unordered_map<mlir::dfcir::utils::Channel *, int32_t> Buffers;
typedef std::unordered_map<mlir::Operation *, unsigned> ModuleArgMap;

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

struct Node {
  Operation *op;
  int32_t latency;

  explicit Node(Operation *op, int32_t latency = -1);

  Node();

  Node(const Node &node) = default;

  ~Node() = default;

  bool operator==(const Node &node) const;
};

struct Channel {
  Node *source;
  Node *target;
  int8_t valInd;
  int32_t offset;

  Channel(Node *source, Node *target,
          int8_t valInd, int32_t offset = 0);

  Channel() = default;

  Channel(const Channel &) = default;

  ~Channel() = default;

  bool operator==(const Channel &channel) const;
};

} // namespace mlir::dfcir::utils

template <>
struct std::hash<mlir::dfcir::utils::Node> {
  using Node = mlir::dfcir::utils::Node;

  size_t operator()(const Node &node) const noexcept {
    return std::hash<mlir::Operation *>()(node.op);
  }
};

template <>
struct std::hash<mlir::dfcir::utils::Channel> {
  using Node = mlir::dfcir::utils::Node;
  using Channel = mlir::dfcir::utils::Channel;

  size_t operator()(const Channel &channel) const noexcept {
    return std::hash<Node *>()(channel.target) + 13 + channel.valInd;
  }
};

namespace mlir::dfcir::utils {

struct NodePtrHash {
  size_t operator()(Node *node) const noexcept {
    return std::hash<Node>()(*node);
  }
};

struct NodePtrEq {
  size_t operator()(Node *left, Node *right) const noexcept {
    return *left == *right;
  }
};

struct ChannelPtrHash {
  size_t operator()(Channel *channel) const noexcept {
    return std::hash<Channel>()(*channel);
  }
};

struct ChannelPtrEq {
  size_t operator()(Channel *left, Channel *right) const noexcept {
    return *left == *right;
  }
};

typedef std::unordered_set<Node *, NodePtrHash, NodePtrEq> Nodes;
typedef std::unordered_set<Channel *, ChannelPtrHash, ChannelPtrEq> Channels;
typedef std::unordered_map<std::string_view, Node *> NodeNameMap;
typedef std::unordered_map<Node *, std::vector<Channel *>> ChannelMap;
typedef std::unordered_map<mlir::detail::ValueImpl *, ConnectOp> ConnectionMap;

class Graph {
public:
  Nodes nodes;
  Channels channels;

  Nodes startNodes;
  ChannelMap inputs;
  ChannelMap outputs;
  ConnectionMap connectionMap;

  explicit Graph() = default;

  ~Graph();

  Node *findNode(Operation *op);

  explicit Graph(KernelOp kernel);

  explicit Graph(ModuleOp module);

  void applyConfig(const LatencyConfig &cfg);

private:
  std::pair<Value, int32_t> findNearestNodeValue(Value value);

  template <class OpGroup, class Op>
  Node *process(Op &op);

  Node *processGenericOp(Operation &op, int32_t latency);
};

void insertBuffer(OpBuilder &builder,
                 Channel *channel,
                 int32_t latency,
                 const ConnectionMap &map);

void insertBuffers(mlir::MLIRContext &ctx,
                   const Buffers &buffers,
                   const ConnectionMap &map);

std::vector<Node *> topSortNodes(const Graph &graph);

int32_t calculateOverallLatency(const Graph &graph,
                                Buffers &buffers,
                                Latencies *map = {});

void eraseOffsets(mlir::Operation *op);

bool hasConstantInput(mlir::Operation *op);

Ops resolveInternalOpType(mlir::Operation *op);

std::string opTypeToString(const Ops &opType);

} // namespace mlir::dfcir::utils

#endif // DFCIR_PASSES_UTILS_H
