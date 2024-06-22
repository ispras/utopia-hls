#include "dfcxx/graph.h"

#include <algorithm>

namespace dfcxx {

Node Graph::findNode(DFVariableImpl *var) {
  return *std::find_if(nodes.begin(), nodes.end(),
                       [&](const Node &node) {
                         return node.var == var;
                       });
}

void Graph::addNode(DFVariableImpl *var, OpType type, NodeData data) {
  nodes.emplace(var, type, data);
  if (type == IN || type == CONST) {
    startNodes.emplace(var, type, data);
  }
}

void Graph::addNode(const DFVariable &var, OpType type, NodeData data) {
  addNode(var.getImpl(), type, data);
}

void Graph::addChannel(DFVariableImpl *source, DFVariableImpl *target, unsigned opInd,
                       bool connect) {
  Node foundSource = findNode(source);
  Node foundTarget = findNode(target);
  Channel newChannel(foundSource, foundTarget, opInd);
  outputs[foundSource].push_back(newChannel);
  inputs[foundTarget].push_back(newChannel);
  if (connect) {
    connections.insert(std::make_pair(foundTarget, newChannel));
    connections.at(foundTarget) = newChannel;
  }
}

void Graph::addChannel(const DFVariable &source, const DFVariable &target, unsigned opInd,
                       bool connect) {
  addChannel(source.getImpl(), target.getImpl(), opInd, connect);
}

GraphHelper::GraphHelper(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
                         KernStorage &storage) : graph(graph), typeBuilder(typeBuilder),
                         varBuilder(varBuilder), storage(storage) {}

void GraphHelper::addNode(DFVariableImpl *var, OpType type, NodeData data) {
  graph.addNode(var, type, data);
}

void GraphHelper::addNode(const DFVariable &var, OpType type, NodeData data) {
  addNode(var.getImpl(), type, data);
}

void GraphHelper::addChannel(DFVariableImpl *source, DFVariableImpl *target, unsigned opInd,
                             bool connect) {
  graph.addChannel(source, target, opInd, connect);
}

void GraphHelper::addChannel(const DFVariable &source, const DFVariable &target, unsigned opInd,
                             bool connect) {
  graph.addChannel(source.getImpl(), target.getImpl(), opInd, connect);
}

} // namespace dfcxx