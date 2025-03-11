//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRSchedulingUtils.h"

namespace mlir::dfcir::utils {

void SchedGraph::applyConfig(const LatencyConfig &cfg) {
  for (SchedNode *node: nodes) {
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

std::pair<Value, int32_t> SchedGraph::findNearestNodeValue(Value value) {
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

SchedNode *SchedGraph::process(Operation *op) {
  // InputOutputOpInterface processing.
  // -------------------------------------------------------
  if (llvm::isa<InputOutputOpInterface>(op)) {
    SchedNode *newNode = new SchedNode(op, 0);
    nodes.insert(newNode);
    if (llvm::isa<InputOpInterface>(op)) {
      inputNodes.insert(newNode);
    }
    if (llvm::isa<OutputOpInterface>(op)) {
      outputNodes.insert(newNode);
    }

    (void)inputs[newNode];
    (void)outputs[newNode];
    return newNode;
  }
  // -------------------------------------------------------
  // ConstantOp processing.
  // -------------------------------------------------------
  if (llvm::isa<ConstantOp>(op)) {
    SchedNode *newNode = new SchedNode(op, 0);
    nodes.insert(newNode);
    inputNodes.insert(newNode);

    (void)inputs[newNode];
    (void)outputs[newNode];
    return newNode;
  }
  // -------------------------------------------------------
  // ConnectOp processing.
  // -------------------------------------------------------
  if (auto casted = llvm::dyn_cast<ConnectOp>(op)) {
    if (!llvm::isa<OutputOpInterface>(casted.getDest().getDefiningOp())) {
      return nullptr;
    }
    auto unrolledInfo = findNearestNodeValue(casted.getSrc());
    SchedNode *srcNode = findNode(unrolledInfo.first.getDefiningOp());
    SchedNode *dstNode = findNode(casted.getDest().getDefiningOp());
  
    auto dstCasted = llvm::cast<OpResult>(casted.getDest());
    int8_t resId = static_cast<int8_t>(dstCasted.getResultNumber());
    int8_t valInd = -resId - 1;
  
    SchedChannel *newChannel = new SchedChannel(srcNode, dstNode, valInd, unrolledInfo.second);
    channels.insert(newChannel);
    outputs[srcNode].push_back(newChannel);
    inputs[dstNode].push_back(newChannel);
  
    return nullptr;
  }
  // -------------------------------------------------------
  // All other operations processing.
  // -------------------------------------------------------
  if (llvm::isa<NaryOpInterface, MuxOp, CastOp,
                       ShiftOpInterface, BitsOp, CatOp>(op)) {
    int32_t latency = llvm::isa<NaryOpInterface>(op) ? -1 : 0;
    SchedNode *newNode = new SchedNode(op, latency);
    nodes.insert(newNode);

    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      auto unrolledInfo = findNearestNodeValue(operand);
      SchedNode *srcNode = findNode(unrolledInfo.first.getDefiningOp());
      SchedChannel *newChannel = new SchedChannel(srcNode, newNode, i, unrolledInfo.second);
      channels.insert(newChannel);
      outputs[srcNode].push_back(newChannel);
      inputs[newNode].push_back(newChannel);
    }

    (void)outputs[newNode];
    return newNode;
  }
  // -------------------------------------------------------

  return nullptr;
}

int32_t calculateOverallLatency(const SchedGraph &graph,
                                SchedGraph::Buffers &buffers,
                                SchedGraph::Latencies *map) {
  bool deleteMap = false;
  int32_t maxLatency = 0;
  
  if (!map) {
    deleteMap = true;
    const std::vector<SchedNode *> sorted = topSortNodes(graph);
    map = new SchedGraph::Latencies();
  
    for (SchedNode *node : sorted) {
      for (SchedChannel *channel : graph.outputs.at(node)) {
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
      SchedChannel *channel = *(std::find_if(ins.begin(),
                                             ins.end(), [] (SchedChannel *ch) {
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

} // namespace mlir::dfcir::utils
