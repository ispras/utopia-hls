//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/conversions/DFCIRPassesUtils.h"

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"



namespace mlir::dfcir::utils {

std::pair<Value, int32_t> findNearestNodeValue(Value value) {
  Value curVal = value;
  int32_t offsetSum = 0;
  bool flag;
  do {
    flag = false;

    for (const auto &operand: curVal.getUses()) {
      auto possConnect = llvm::dyn_cast<ConnectOp>(operand.getOwner());
      if (!possConnect) { continue; }
      if (possConnect.getDest() == operand.get()) {
        curVal = possConnect.getSrc();
        flag = true;
        break;
      }
    }
    
    if (!flag) {
      auto possOffset = llvm::dyn_cast<OffsetOp>(curVal.getDefiningOp());
      if (possOffset) {
        offsetSum += static_cast<int32_t>(possOffset.getOffset().getInt());
        curVal = possOffset.getStream();
        flag = true;
      }
    }
  
  } while (flag);

  return std::make_pair(curVal, offsetSum);
}

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

auto Graph::findNode(Operation *op) {
  return std::find_if(nodes.begin(), nodes.end(),
      [&](Node *n) {
        return n->op == op;
      });
}

auto Graph::findNode(const Value &val) {
  return std::find_if(nodes.begin(), nodes.end(),
      [&](Node *n) {
        return n->op == val.getDefiningOp();
      });
}

void Graph::applyConfig(const LatencyConfig &cfg) {
  for (Node *node: nodes) {
    auto casted = llvm::dyn_cast<Scheduled>(node->op);
    if (!casted) { continue; }

    Ops opType = resolveInternalOpType(node->op);

    auto found = cfg.internalOps.find(opType);

    int32_t latency = (found != cfg.internalOps.end()) ?
                      (*found).second :
                      1;
    
    casted.setLatency(latency);

    node->latency = latency;
  }
}

template <>
void Graph::process<InputOutputOpInterface>(InputOutputOpInterface &op) {
  Node *newNode = new Node(op, 0);
  nodes.insert(newNode);
  if (llvm::isa<InputOpInterface>(op.getOperation())) {
    startNodes.insert(newNode);
  }
}

template <>
void Graph::process<ConstantOp>(ConstantOp &op) {
  Node *newNode = new Node(op, 0);
  nodes.insert(newNode);
  startNodes.insert(newNode);
}

template <>
void Graph::process<MuxOp>(MuxOp &op) {
  Node *newNode = new Node(op, 0);
  nodes.insert(newNode);

  for (size_t i = 0; i < op.getNumOperands(); ++i) {
    auto operand = op.getOperand(i);
    auto unrolledInfo = findNearestNodeValue(operand);
    auto srcNode = findNode(unrolledInfo.first);
    Channel *newChannel = new Channel(*srcNode, newNode, i, unrolledInfo.second);
    channels.insert(newChannel);
    outputs[*srcNode].insert(newChannel);
    inputs[newNode].insert(newChannel);
  }
}

template <>
void Graph::process<ConnectOp>(ConnectOp &op) {
  if (!llvm::isa<OutputOpInterface>(op.getDest().getDefiningOp())) { return; }
  auto unrolledInfo = findNearestNodeValue(op.getSrc());
  auto srcNode = findNode(unrolledInfo.first);
  auto dstNode = findNode(op.getDest());
  Channel *newChannel = new Channel(*srcNode, *dstNode, 0, unrolledInfo.second);
  channels.insert(newChannel);
  outputs[*srcNode].insert(newChannel);
  inputs[*dstNode].insert(newChannel);
}

template <>
void Graph::process<NaryOpInterface>(NaryOpInterface &op) {
  Node *newNode = new Node(op, -1);
  nodes.insert(newNode);

  Operation *opPtr = op.getOperation();

  for (size_t i = 0; i < opPtr->getNumOperands(); ++i) {
    auto operand = opPtr->getOperand(i);
    auto unrolledInfo = findNearestNodeValue(operand);
    auto srcNode = findNode(unrolledInfo.first);
    Channel *newChannel = new Channel(*srcNode, newNode, i, unrolledInfo.second);
    channels.insert(newChannel);
    outputs[*srcNode].insert(newChannel);
    inputs[newNode].insert(newChannel);
  }
}

template <>
void Graph::process<ShiftOpInterface>(ShiftOpInterface &op) {
  Node *newNode = new Node(op, 0);
  nodes.insert(newNode);

  Operation *opPtr = op.getOperation();

  auto operand = opPtr->getOperand(0);
  auto unrolledInfo = findNearestNodeValue(operand);
  auto srcNode = findNode(unrolledInfo.first);
  Channel *newChannel = new Channel(*srcNode, newNode, 0, unrolledInfo.second);
  outputs[*srcNode].insert(newChannel);
  inputs[newNode].insert(newChannel);
}

Graph::Graph(KernelOp kernel) : Graph() {

  for (Operation &op: kernel.getBody().front().getOperations()) {
    if (auto casted = llvm::dyn_cast<InputOutputOpInterface>(&op)) {
      process<InputOutputOpInterface>(casted);
    } else if (auto casted = llvm::dyn_cast<ConstantOp>(&op)) {
      process<ConstantOp>(casted);
    } else if (auto casted = llvm::dyn_cast<MuxOp>(&op)) {
      process<MuxOp>(casted);
    } else if (auto casted = llvm::dyn_cast<ConnectOp>(&op)) {
      process<ConnectOp>(casted);
    } else if (auto casted = llvm::dyn_cast<NaryOpInterface>(&op)) {
      process<NaryOpInterface>(casted);
    } else if (auto casted = llvm::dyn_cast<ShiftOpInterface>(&op)) {
      process<ShiftOpInterface>(casted);
    }
  }
}

Graph::Graph(ModuleOp module)
    : Graph(mlir::utils::findFirstOccurence<KernelOp>(module)) {}


void insertBuffer(OpBuilder &builder, Channel *channel, int32_t latency) {

  builder.setInsertionPoint(channel->target->op);
  auto value = channel->target->op->getOperand(channel->valInd);

  auto latencyOp = builder.create<LatencyOp>(builder.getUnknownLoc(),
                                             value.getType(),
                                             value,
                                             latency);
  channel->target->op->setOperand(channel->valInd, latencyOp.getRes());
}

void insertBuffers(mlir::MLIRContext &ctx, const Buffers &buffers) {
  OpBuilder builder(&ctx);
  for (auto &[channel, latency]: buffers) {
    insertBuffer(builder, channel, latency);
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

  assert(false && "Shouldn't reach this");
  return Ops::UNDEFINED;
}

} // namespace mlir::dfcir::utils