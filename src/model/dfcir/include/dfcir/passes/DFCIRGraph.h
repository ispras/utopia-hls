//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCIR_GRAPH_H
#define DFCIR_GRAPH_H

#include "dfcir/passes/DFCIRPasses.h"
#include "dfcir/DFCIROperations.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlir::utils {

template <typename OpTy>
inline OpTy findFirstOccurence(Operation *op);

} // namespace mlir::utils

namespace mlir::dfcir::utils {

typedef uint64_t NodeID;

template <class ChannelClass>
struct GraphNode {
  NodeID id;
  Operation *op;
  std::vector<ChannelClass *> inputs;
  std::vector<ChannelClass *> outputs;

  GraphNode() = default;

  GraphNode(const GraphNode &) = default;

  GraphNode(NodeID id, Operation *op) : inputs(), outputs() {
    this->id = id;
    this->op = op;
  }

  ~GraphNode() = default;

  bool operator==(const GraphNode &node) const {
    return this->op == node.op;
  }
};

template <class NodeClass>
struct GraphChannel {
  NodeClass *source;
  NodeClass *target;
  int8_t valInd;

  GraphChannel() = default;

  GraphChannel(const GraphChannel &) = default;

  GraphChannel(NodeClass *source, NodeClass *target, int8_t valInd) {
    this->source = source;
    this->target = target;
    this->valInd = valInd;
  }

  ~GraphChannel() = default;

  bool operator==(const GraphChannel &channel) const {
    return this->source == channel.source &&
           this->target == channel.target &&
           this->valInd == channel.valInd;
  }
};

template <class NodeClass, class ChannelClass>
class Graph {
public:
  typedef std::vector<NodeClass *> Nodes;
  typedef std::vector<ChannelClass *> Channels;
  typedef std::unordered_set<NodeClass *> NodesSet;
  typedef std::unordered_map<mlir::detail::ValueImpl *,
                             ConnectOp> ConnectionMap;
  typedef std::unordered_map<Operation *, NodeClass *> OpNodeMap;
  typedef std::unordered_map<ChannelClass *, int32_t> Buffers;

  Nodes nodes;
  Channels channels;

  Nodes inputNodes;
  Nodes outputNodes;
  ConnectionMap connectionMap;

  Graph() = default;

  virtual ~Graph() {
    for (NodeClass *node: nodes) {
      delete node;
    }
  
    for (ChannelClass *channel: channels) {
      delete channel;
    }
  }

  inline bool isConstantInput(NodeClass *node) const {
    return llvm::isa<ConstantInputInterface>(node->op);
  }

  inline bool isInput(NodeClass *node) const {
    return llvm::isa<InputOpInterface>(node->op);
  }

  inline bool isOutput(NodeClass *node) const {
    return llvm::isa<OutputOpInterface>(node->op);
  }

  NodeClass *findNode(Operation *op, const OpNodeMap &map) {
    auto found = map.find(op);
    if (found != map.end()) {
      return found->second;
    }
    return nullptr;
  }

  void constructFrom(KernelOp kernel) {
    Block &block = kernel.getBody().front();

    for (ConnectOp connect: block.getOps<ConnectOp>()) {
      connectionMap[connect.getDest().getImpl()] = connect;
    }

    OpNodeMap map;

    for (Operation &op: block.getOperations()) {
      process(&op, map);
    }
  }

  void constructFrom(ModuleOp module) {
    constructFrom(mlir::utils::findFirstOccurence<KernelOp>(module));
  }

  void insertBuffer(OpBuilder &builder, ChannelClass *channel, int32_t latency) {
    if (latency <= 0) {
      std::cout << "Created a buffer with latency <= 0 (" << latency;
      std::cout << ')' << std::endl << "between";
      channel->source->op->dump();
      std::cout << "and";
      channel->target->op->dump();
      assert (latency > 0);
    }
  
    if (channel->valInd >= 0) {
      builder.setInsertionPoint(channel->target->op);
      auto value = channel->target->op->getOperand(channel->valInd);
  
      auto latencyOp = builder.create<LatencyOp>(builder.getUnknownLoc(),
                                                 value.getType(),
                                                 value,
                                                 latency);
      channel->target->op->setOperand(channel->valInd, latencyOp.getRes());
    } else {
      unsigned resId = static_cast<unsigned>(-(channel->valInd + 1));
      auto targetRes = channel->target->op->getResult(resId);
      auto foundConnect = connectionMap.find(targetRes.Value::getImpl());
      assert(foundConnect != connectionMap.end());
      ConnectOp connect = (*foundConnect).second;
      builder.setInsertionPoint(connect);
      auto connectSrc = connect.getSrc();
      auto latencyOp = builder.create<LatencyOp>(builder.getUnknownLoc(),
                                                 connectSrc.getType(),
                                                 connectSrc,
                                                 latency);
      connect.getSrcMutable().assign(latencyOp.getRes());
    }
  }

  void insertBuffers(mlir::MLIRContext &ctx, const Buffers &buffers) {
    OpBuilder builder(&ctx);
    for (auto &[channel, latency]: buffers) {
      insertBuffer(builder, channel, latency);
    }
  }

protected:
  virtual NodeClass *process(Operation *op, OpNodeMap &map) = 0;
};

template <class NodeClass, class ChannelClass>
std::vector<NodeClass *> topSortNodes(const Graph<NodeClass, ChannelClass> &graph) {
  size_t nodesCount = graph.nodes.size();
  auto &outs = graph.outputs;

  std::vector<NodeClass *> result(nodesCount);

  std::unordered_map<NodeClass *, size_t> checked;
  std::stack<NodeClass *> stack;

  for (NodeClass *node: graph.inputNodes) {
    stack.push(node);
    checked[node] = 0;
  }

  size_t i = nodesCount;
  while (!stack.empty() && i > 0) {
    NodeClass *node = stack.top();
    size_t count = outs.at(node).size();
    size_t curr;
    bool flag = true;
    for (curr = checked[node]; flag && curr < count; ++curr) {
      ChannelClass *next = outs.at(node)[curr];
      if (checked.find(next->target) == checked.end()) {
        checked[next->target] = 0;
        stack.push(next->target);
        flag = false;
      }
      ++checked[node];
    }

    if (flag) {
      stack.pop();
      result[--i] = node;
    }
  }
  assert(stack.empty());
  assert(i == 0);
  return result;
}

} // namespace mlir::dfcir::utils

#endif // DFCIR_GRAPH_H
