//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcir/passes/DFCIRPipeliningUtils.h"

namespace mlir::dfcir::utils {

Value CombGraph::findNearestNodeValue(Value value) {
  Value curVal = value;
  auto foundConnect = connectionMap.find(curVal.getImpl());
  while (foundConnect != connectionMap.end()) {
    curVal = foundConnect->second.getSrc();
  }

  return curVal;
}

CombNode *CombGraph::process(Operation *op, OpNodeMap &map) {
  NodeID newId = nodes.size();

  // InputOutputOpInterface processing.
  // -------------------------------------------------------
  if (llvm::isa<InputOutputOpInterface>(op)) {
    CombNode *newNode = new CombNode(newId, op, 0.f);
    nodes.push_back(newNode);
    map[op] = newNode;

    if (llvm::isa<InputOpInterface>(op)) {
      inputNodes.push_back(newNode);
    }
    if (llvm::isa<OutputOpInterface>(op)) {
      outputNodes.push_back(newNode);
    }

    return newNode;
  }
  // -------------------------------------------------------
  // ConstantOp processing.
  // -------------------------------------------------------
  if (llvm::isa<ConstantOp>(op)) {
    CombNode *newNode = new CombNode(newId, op, 0.f);
    nodes.push_back(newNode);
    inputNodes.push_back(newNode);
    map[op] = newNode;

    return newNode;
  }
  // -------------------------------------------------------
  // ConnectOp processing.
  // -------------------------------------------------------
  if (auto casted = llvm::dyn_cast<ConnectOp>(op)) {
    if (!llvm::isa<OutputOpInterface>(casted.getDest().getDefiningOp())) {
      return nullptr;
    }
    auto foundValue = findNearestNodeValue(casted.getSrc());
    CombNode *srcNode = findNode(foundValue.getDefiningOp(), map);
    CombNode *dstNode = findNode(casted.getDest().getDefiningOp(), map);
  
    auto dstCasted = llvm::cast<OpResult>(casted.getDest());
    int8_t resId = static_cast<int8_t>(dstCasted.getResultNumber());
    int8_t valInd = -resId - 1;
  
    CombChannel *newChannel = new CombChannel(srcNode, dstNode, valInd);
    channels.push_back(newChannel);
    srcNode->outputs.push_back(newChannel);
    dstNode->inputs.push_back(newChannel);
  
    return nullptr;
  }
  // -------------------------------------------------------
  // All other operations processing.
  // -------------------------------------------------------
  if (llvm::isa<NaryOpInterface, MuxOp, CastOp,
                       ShiftOpInterface, BitsOp, CatOp>(op)) {
    CombLatency latency = 1.f;
    CombNode *newNode = new CombNode(newId, op, latency);
    nodes.push_back(newNode);
    map[op] = newNode;

    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      auto operand = op->getOperand(i);
      auto foundValue = findNearestNodeValue(operand);
      CombNode *srcNode = findNode(foundValue.getDefiningOp(), map);
      CombChannel *newChannel = new CombChannel(srcNode, newNode, i);
      channels.push_back(newChannel);
      srcNode->outputs.push_back(newChannel);
      newNode->inputs.push_back(newChannel);
    }

    return newNode;
  }
  // -------------------------------------------------------

  return nullptr;
}

void CombGraph::divideIntoLayers(NodeLayers &nodeLayers,
                                 LayerLatencies &layerLatencies) {
  // There needs to be a layer for each node.
  // Explicitly assign the first layer (0) to every node.
  nodeLayers.resize(nodes.size(), 0);

  uint64_t nodesCount = nodes.size();
  uint64_t maxLayerId = 0;

  // Visit every node in topological order and update
  // each output's assigned layer.
  for (uint64_t i = 0; i < nodesCount; ++i) {
    CombNode *node = nodes[i];
    uint64_t newLayerId = nodeLayers[i] + 1;

    // Update the maximum layer ID on demand.
    if (!node->outputs.empty() && maxLayerId < newLayerId) {
      maxLayerId = newLayerId;
    }

    for (CombChannel *channel: node->outputs) {
      CombNode *targetNode = channel->target;
      NodeID targetId = targetNode->id;
      if (nodeLayers[targetId] < newLayerId) {
        nodeLayers[targetId] = newLayerId;
      }
    }
  }

  // Allocate space for every layer's weight.
  layerLatencies.resize(maxLayerId + 1, 0);

  // Compute the weight for each layer.
  for (uint64_t i = 0; i < nodesCount; ++i) {
    CombNode *node = nodes[i];
    uint64_t layerId = nodeLayers[i];
    if (layerLatencies[layerId] < node->latency) {
      layerLatencies[layerId] = node->latency;
    }
  }
}
  
void CombGraph::divideIntoCascades(const uint64_t cascadesCount,
                                   const LayerLatencies &layerLatencies,
                                   LayerCascades &layerCascades,
                                   CascadeBoundaries &cascadeBoundaries) {
  assert(cascadesCount > 0);
  // There needs to be a cascade for each layer.
  // Explicitly assign the first cascade (0) to every layer.
  layerCascades.resize(layerLatencies.size(), 0);
  // There needs to be a rightmost element for every cascade.
  // In case there are less layers than cascades,
  // the last layer is added to the rest of the cascades.
  cascadeBoundaries.resize(cascadesCount, 0);

  CombLatency latSum = 0;

  for (CombLatency latency: layerLatencies) {
    latSum += latency;
  }

  uint64_t leftCascades = cascadesCount;
  CombLatency latAvg = latSum / static_cast<CombLatency>(leftCascades);
  CombCascadeID currCascade = 0;
  CombLatency currSum = 0.f;
  CombLayerID i = 0;
  CombLayerID layerCount = layerLatencies.size();

  for (; i < layerCount && leftCascades > 1; ++i) {
    CombLatency oldSum = currSum;
    currSum += layerLatencies[i];

    if (latAvg - currSum > floatEps) {
      layerCascades[i] = currCascade;
      continue;
    }

    if (currSum - latAvg - latAvg + oldSum > floatEps) {
      assert(i > 0); // TODO: Check for possible cases.
      cascadeBoundaries[currCascade] = i - 1;
      currSum = layerLatencies[i];
      ++currCascade;
      layerCascades[i] = currCascade;
    } else {
      cascadeBoundaries[currCascade] = i;
      oldSum = currSum;
      currSum = 0.f;
      layerCascades[i] = currCascade;
      ++currCascade;
    }

    --leftCascades;
    latSum -= oldSum;
    latAvg = latSum / static_cast<CombLatency>(leftCascades);
  }

  // If there are unassigned layers left - put them into the last cascade.
  if (i < layerCount) {
    for (; i < layerCount; ++i) {
      layerCascades[i] = cascadesCount - 1;
    }
    cascadeBoundaries[cascadesCount - 1] = layerCount - 1;
  }
  // If there are unassigned cascades left - explicitly put the last layer
  // in the last cascade.
  if (i == layerCount && leftCascades > 0) {
    layerCascades[layerCount - 1] = cascadesCount - 1;
    cascadeBoundaries[cascadesCount - 1] = layerCount - 1;
  }
}

CombGraph::Buffers CombGraph::calculateFIFOs(const NodeLayers &nodeLayers,
                                             const LayerCascades &layersCascades,
                                             const CascadeBoundaries &cascadeBoundaries) {
  return {};
}

} // namespace mlir::dfcir::utils
