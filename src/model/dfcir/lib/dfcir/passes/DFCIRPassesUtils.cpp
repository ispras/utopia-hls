//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPassesUtils.h"

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <stack>

namespace mlir::dfcir::utils {

Node::Node(Operation *op, int32_t latency) : op(op), latency(latency) {}

Node::Node() : Node(nullptr) {}

bool Node::operator==(const Node &node) const {
  return this->op == node.op;
}

Channel::Channel(Node *source, Node *target,
                 int8_t valInd, int32_t offset) :
                 source(source), target(target),
                 valInd(valInd), offset(offset) {}

bool Channel::operator==(const Channel &channel) const {
  return this->source == channel.source &&
         this->target == channel.target &&
         this->valInd == channel.valInd;
}

Graph::~Graph() {
  for (Node *node: nodes) {
    delete node;
  }

  for (Channel *channel: channels) {
    delete channel;
  }
}

Node *Graph::findNode(Operation *op) {
  Node *bufNode = new Node(op);
  auto found = nodes.find(bufNode);
  delete bufNode;
  if (found != nodes.end()) {
    return *found;
  }
  return nullptr;
}

void Graph::applyConfig(const LatencyConfig &cfg) {
  for (Node *node: nodes) {
    auto casted = llvm::dyn_cast<Scheduled>(node->op);
    if (!casted) { continue; }

    Ops opType = resolveInternalOpType(node->op);

    if (opType == Ops::UNDEFINED) {
      std::cout << "Couldn't deduce the type for the operation below. ";
      std::cout << "Latency 0 will be used." << std::endl;
      node->op->dump();
    }

    auto found = cfg.internalOps.find(opType);

    int32_t latency;
    if (found != cfg.internalOps.end()) {
      latency = (*found).second;
    } else {
      std::cout << "No explicit config for operation type "
                << opTypeToString(opType)
                << "." << std::endl << "Latency 1 will be used." << std::endl;
      latency = 1;
    }
    
    casted.setLatency(latency);

    node->latency = latency;
  }
}

std::pair<Value, int32_t> Graph::findNearestNodeValue(Value value) {
  Value curVal = value;
  int32_t offsetSum = 0;
  bool flag;
  do {
    flag = false;

    auto possOffset = llvm::dyn_cast<OffsetOp>(curVal.getDefiningOp());
    if (possOffset) {
      offsetSum += static_cast<int32_t>(possOffset.getOffset().getInt());
      curVal = possOffset.getStream();
      flag = true;
    }

    auto foundConnect = connectionMap.find(curVal.getImpl());
    if (foundConnect != connectionMap.end()) {
      curVal = (*foundConnect).second.getSrc();
      flag = true;
    }

  } while (flag);

  return std::make_pair(curVal, offsetSum);
}

template <>
Node *Graph::process<InputOutputOpInterface>(InputOutputOpInterface &op) {
  Node *newNode = new Node(op, 0);
  nodes.insert(newNode);
  if (llvm::isa<InputOpInterface>(op.getOperation())) {
    startNodes.insert(newNode);
  }

  (void)inputs[newNode];
  (void)outputs[newNode];
  return newNode;
}

template <>
Node *Graph::process<ConstantOp>(ConstantOp &op) {
  Node *newNode = new Node(op, 0);
  nodes.insert(newNode);
  startNodes.insert(newNode);

  (void)inputs[newNode];
  (void)outputs[newNode];
  return newNode;
}

template <>
Node *Graph::process<ConnectOp>(ConnectOp &op) {
  if (!llvm::isa<OutputOpInterface>(op.getDest().getDefiningOp())) {
    return nullptr;
  }
  auto unrolledInfo = findNearestNodeValue(op.getSrc());
  Node *srcNode = findNode(unrolledInfo.first.getDefiningOp());
  Node *dstNode = findNode(op.getDest().getDefiningOp());

  auto dstCasted = llvm::cast<OpResult>(op.getDest());
  int8_t resId = static_cast<int8_t>(dstCasted.getResultNumber());
  int8_t valInd = -resId - 1;

  Channel *newChannel = new Channel(srcNode, dstNode, valInd, unrolledInfo.second);
  channels.insert(newChannel);
  outputs[srcNode].push_back(newChannel);
  inputs[dstNode].push_back(newChannel);

  return nullptr;
}

Node *Graph::processGenericOp(Operation &op, int32_t latency) {
  Node *newNode = new Node(&op, latency);
  nodes.insert(newNode);

  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    auto operand = op.getOperand(i);
    auto unrolledInfo = findNearestNodeValue(operand);
    Node *srcNode = findNode(unrolledInfo.first.getDefiningOp());
    Channel *newChannel = new Channel(srcNode, newNode, i, unrolledInfo.second);
    channels.insert(newChannel);
    outputs[srcNode].push_back(newChannel);
    inputs[newNode].push_back(newChannel);
  }

  (void)outputs[newNode];
  return newNode;
}

Graph::Graph(KernelOp kernel) : Graph() {
  Block &block = kernel.getBody().front();

  for (ConnectOp connect: block.getOps<ConnectOp>()) {
    connectionMap[connect.getDest().getImpl()] = connect;
  }

  for (Operation &op: block.getOperations()) {
    if (auto casted = llvm::dyn_cast<InputOutputOpInterface>(&op)) {
      process<InputOutputOpInterface>(casted);
    } else if (auto casted = llvm::dyn_cast<ConstantOp>(&op)) {
      process<ConstantOp>(casted);
    } else if (auto casted = llvm::dyn_cast<ConnectOp>(&op)) {
      process<ConnectOp>(casted);
    } else if (llvm::isa<NaryOpInterface>(&op)) {
      processGenericOp(op, -1);
    } else if (llvm::isa<MuxOp, CastOp, ShiftOpInterface,
                         BitsOp, CatOp>(&op)) {
      processGenericOp(op, 0);
    }
  }
}

Graph::Graph(ModuleOp module)
    : Graph(mlir::utils::findFirstOccurence<KernelOp>(module)) {}

void insertBuffer(OpBuilder &builder, Channel *channel,
                  int32_t latency,const ConnectionMap &map) {
  if (latency <= 0) {
    std::cout << "Scheduling created a buffer with latency <= 0 (" << latency;
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
    auto foundConnect = map.find(targetRes.Value::getImpl());;
    assert(foundConnect != map.end());
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

void insertBuffers(mlir::MLIRContext &ctx, const Buffers &buffers,
                   const ConnectionMap &map) {
  OpBuilder builder(&ctx);
  for (auto &[channel, latency]: buffers) {
    insertBuffer(builder, channel, latency, map);
  }
}

void eraseOffsets(mlir::Operation *op) {
  op->walk([](OffsetOp offset) {
    auto input = offset->getOperand(0);
    auto result = offset->getResult(0);
    for (auto &operand: llvm::make_early_inc_range(result.getUses())) {
      operand.set(input);
    }
    offset->erase();
  });
}

std::vector<Node *> topSortNodes(const Graph &graph) {
  size_t nodesCount = graph.nodes.size();
  auto &outs = graph.outputs;

  std::vector<Node *> result(nodesCount);

  std::unordered_map<Node *, size_t> checked;
  std::stack<Node *> stack;

  for (Node *node: graph.startNodes) {
    stack.push(node);
    checked[node] = 0;
  }

  size_t i = nodesCount;
  while (!stack.empty() && i > 0) {
    Node *node = stack.top();
    size_t count = outs.at(node).size();
    size_t curr;
    bool flag = true;
    for (curr = checked[node]; flag && curr < count; ++curr) {
      Channel *next = outs.at(node)[curr];
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

int32_t calculateOverallLatency(const Graph &graph, Buffers &buffers, Latencies *map) {
  bool deleteMap = false;
  int32_t maxLatency = 0;

  if (!map) {
    deleteMap = true;
    const std::vector<Node *> sorted = topSortNodes(graph);
    map = new Latencies();

    for (Node *node : sorted) {
      for (Channel *channel : graph.outputs.at(node)) {
        int32_t latency = (*map)[node] + node->latency + channel->offset;
        auto foundBuf = buffers.find(channel);
        if (foundBuf != buffers.end()) {
          latency += foundBuf->second;
        }

        if (latency > (*map)[channel->target]) {
          (*map)[channel->target] = latency;
        }

        if (llvm::isa<OutputOpInterface>(channel->target->op) &&
            latency > maxLatency) {
          maxLatency = latency;
        }
      }
    }
  } else {
    for (const auto &[node, latency] : *map) {
      if (llvm::isa<OutputOpInterface>(node->op) &&
          latency > maxLatency) {
        maxLatency = latency;
      }
    }
  }

  for (auto &[node, latency] : *map) {
    if (llvm::isa<OutputOpInterface>(node->op)) {
      const auto &ins = graph.inputs.at(node);
      Channel *channel = *(std::find_if(ins.begin(),
                                        ins.end(), [] (Channel *ch) {
        return ch->valInd == -1;
      }));

      int32_t delta = latency - (*map)[channel->source];
      auto foundBuf = buffers.find(channel);
      if (foundBuf != buffers.end()) {
        delta -= foundBuf->second;
      }

      if (delta > 0) {
        buffers[channel] = delta;
      }
    }
  }

  if (deleteMap) {
    delete map;
  }

  return maxLatency;
}

bool hasConstantInput(mlir::Operation *op) {
  return llvm::isa<ConstantInputInterface>(op);
}

Ops resolveInternalOpType(mlir::Operation *op) {
  auto resultType = op->getResult(0).getType();
  auto dfType = llvm::dyn_cast<DFType>(resultType).getDFType();
  bool isFloat = llvm::isa<DFCIRFloatType>(dfType);
 
  if (llvm::isa<AddOp>(op)) {
    return (isFloat) ? Ops::ADD_FLOAT : Ops::ADD_INT;
  } else if (llvm::isa<SubOp>(op)) {
    return (isFloat) ? Ops::SUB_FLOAT : Ops::SUB_INT;
  } else if (llvm::isa<MulOp>(op)) {
    return (isFloat) ? Ops::MUL_FLOAT : Ops::MUL_INT;
  } else if (llvm::isa<DivOp>(op)) {
    return (isFloat) ? Ops::DIV_FLOAT : Ops::DIV_INT;
  } else if (llvm::isa<NegOp>(op)) {
    return (isFloat) ? Ops::NEG_FLOAT : Ops::NEG_INT;
  } else if (llvm::isa<AndOp>(op)) {
    return (isFloat) ? Ops::AND_FLOAT : Ops::AND_INT;
  } else if (llvm::isa<OrOp>(op)) {
    return (isFloat) ? Ops::OR_FLOAT : Ops::OR_INT;
  } else if (llvm::isa<XorOp>(op)) {
    return (isFloat) ? Ops::XOR_FLOAT : Ops::XOR_INT;
  } else if (llvm::isa<NotOp>(op)) {
    return (isFloat) ? Ops::NOT_FLOAT : Ops::NOT_INT;
  } else if (llvm::isa<LessOp>(op)) {
    return (isFloat) ? Ops::LESS_FLOAT : Ops::LESS_INT;
  } else if (llvm::isa<LessEqOp>(op)) {
    return (isFloat) ? Ops::LESSEQ_FLOAT : Ops::LESSEQ_INT;
  } else if (llvm::isa<GreaterOp>(op)) {
    return (isFloat) ? Ops::GREATER_FLOAT : Ops::GREATER_INT;
  } else if (llvm::isa<GreaterEqOp>(op)) {
    return (isFloat) ? Ops::GREATEREQ_FLOAT : Ops::GREATEREQ_INT;
  } else if (llvm::isa<EqOp>(op)) {
    return (isFloat) ? Ops::EQ_FLOAT : Ops::EQ_INT;
  } else if (llvm::isa<NotEqOp>(op)) {
    return (isFloat) ? Ops::NEQ_FLOAT : Ops::NEQ_INT;
  }

  return Ops::UNDEFINED;
}

std::string opTypeToString(const Ops &opType) {
  switch (opType) {
  case ADD_INT: return "ADD_INT";
  case ADD_FLOAT: return "ADD_FLOAT";
  case SUB_INT: return "SUB_INT";
  case SUB_FLOAT: return "SUB_FLOAT";
  case MUL_INT: return "MUL_INT";
  case MUL_FLOAT: return "MUL_FLOAT";
  case DIV_INT: return "DIV_INT";
  case DIV_FLOAT: return "DIV_FLOAT";
  case NEG_INT: return "NEG_INT";
  case NEG_FLOAT: return "NEG_FLOAT";
  case AND_INT: return "AND_INT";
  case AND_FLOAT: return "AND_FLOAT";
  case OR_INT: return "OR_INT";
  case OR_FLOAT: return "OR_FLOAT";
  case XOR_INT: return "XOR_INT";
  case XOR_FLOAT: return "XOR_FLOAT";
  case NOT_INT: return "NOT_INT";
  case NOT_FLOAT: return "NOT_FLOAT";
  case LESS_INT: return "LESS_INT";
  case LESS_FLOAT: return "LESS_FLOAT";
  case LESSEQ_INT: return "LESSEQ_INT";
  case LESSEQ_FLOAT: return "LESSEQ_FLOAT";
  case GREATER_INT: return "GREATER_INT";
  case GREATER_FLOAT: return "GREATER_FLOAT";
  case GREATEREQ_INT: return "GREATEREQ_INT";
  case GREATEREQ_FLOAT: return "GREATEREQ_FLOAT";
  case EQ_INT: return "EQ_INT";
  case EQ_FLOAT: return "EQ_FLOAT";
  case NEQ_INT: return "NEQ_INT";
  case NEQ_FLOAT: return "NEQ_FLOAT";
  default: return "<unknown>";
  }
}

} // namespace mlir::dfcir::utils
