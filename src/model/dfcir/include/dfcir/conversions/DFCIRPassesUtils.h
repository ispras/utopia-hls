//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_PASSES_UTILS_H
#define DFCIR_PASSES_UTILS_H

#include "dfcir/conversions/DFCIRPasses.h"
#include "dfcir/DFCIROperations.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace mlir::dfcir::utils {
struct Node;
struct Channel;
struct Graph;
} // namespace mlir::dfcir::utils

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
class Graph {
  using StringRef = llvm::StringRef;

public:
  std::unordered_set<Node *> nodes;
  std::unordered_set<Channel *> channels;

  std::unordered_set<Node *> startNodes;
  std::unordered_map<Node *, std::unordered_set<Channel *>> inputs;
  std::unordered_map<Node* , std::unordered_set<Channel *>> outputs;

  explicit Graph() = default;

  ~Graph();

  auto findNode(Operation *op);

  auto findNode(const Value &val);

  explicit Graph(KernelOp kernel);

  explicit Graph(ModuleOp module);

  void applyConfig(const LatencyConfig &cfg);

private:
  template <class OpGroup, class Op>
  Node *process(Op &op);

  Node *processGenericOp(Operation &op, int32_t latency);
};

void insertBuffer(OpBuilder &builder, Channel *channel, int32_t latency);

void insertBuffers(mlir::MLIRContext &ctx, const Buffers &buffers);

int32_t calculateOverallLatency(const Graph &graph, const Buffers &buffers);

void eraseOffsets(mlir::Operation *op);

bool hasConstantInput(mlir::Operation *op);

Ops resolveInternalOpType(mlir::Operation *op);

std::string opTypeToString(const Ops &opType);

} // namespace mlir::dfcir::utils

#endif // DFCIR_PASSES_UTILS_H
