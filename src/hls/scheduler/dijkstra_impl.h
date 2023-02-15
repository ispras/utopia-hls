//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

template<>
inline GenericChanQueue::Queue(CompareChan comparator) {
  container = new StdPriorityQueue(comparator);
}

template<>
inline const model::Chan* GenericChanQueue::front() {
    return container->top();
}

template <typename Container, typename Comparator>
void DijkstraBalancer<Container, Comparator>::initQueue() {
  delete toVisit;
  toVisit = new Queue<const model::Chan*, Container, Comparator>();
}

template <>
inline void DijkstraBalancer<StdPriorityQueue, CompareChan>::initQueue() {
  delete toVisit;
  toVisit = new GenericChanQueue(CompareChan(nodeMap));
}

template <typename Container, typename Comparator>
void DijkstraBalancer<Container, Comparator>::init(const model::Graph *graph) {
  TraverseBalancerBase::init(*graph);
  initQueue();
  currentNode = nullptr;
}

template <typename Container, typename Comparator>
void DijkstraBalancer<Container, Comparator>::visitChan(
    const model::Chan *chan) {
  const Node *targetNode = getNextNode(chan);
  // Update destination node time
  if (targetNode && currentNode) {
    nodeMap[targetNode] = std::max(nodeMap[currentNode] + (int)chan->ind.ticks, 
      nodeMap[targetNode]);
    graphTime = std::max((unsigned)nodeMap[targetNode], graphTime);
  }
}

template <typename Container, typename Comparator>
void DijkstraBalancer<Container, Comparator>::visitNode(
    const model::Node *node) {
  currentNode = node;
  for (const auto *chan : getConnections(node)) {
    toVisit->push(chan);
    visitChan(chan);
  }
}

template <typename Container, typename Comparator>
void DijkstraBalancer<Container, Comparator>::start(
    const std::vector<model::Node*> &startNodes) {
  for (const auto *node : startNodes) {
    visitNode(node);
  }
}

template <typename Container, typename Comparator>
const model::Node* DijkstraBalancer<Container, Comparator>::getNextNode(
    const model::Chan *chan) {
  return 
    mode == LatencyBalanceMode::ASAP ? chan->target.node : chan->source.node;
}

template <typename Container, typename Comparator>
const std::vector<model::Chan*>& DijkstraBalancer<Container, Comparator>
    ::getConnections(const model::Node *next) {
  return mode == LatencyBalanceMode::ASAP ? next->outputs : next->inputs;
}

template <typename Container, typename Comparator>
void DijkstraBalancer<Container, Comparator>::traverse(
    const std::vector<model::Node*> &startNodes) {
  start(startNodes);
  while (!toVisit->empty()) {
    const Node *next = getNextNode(toVisit->front());
    toVisit->pop();
    visitNode(next);
  }
}

template <typename Container, typename Comparator>
void DijkstraBalancer<Container, Comparator>::balance(model::Model &model, 
    LatencyBalanceMode balanceMode) {
  mode = balanceMode;
  const model::Graph *graph = model.main();
  uassert(graph, "Graph 'main' not found!\n");
  init(graph);

  if (mode == LatencyBalanceMode::ASAP) {
    traverse(graph->sources);
  }

  if (mode == LatencyBalanceMode::ALAP) {
    traverse(graph->targets);
  }

  insertBuffers(model);
  printGraphTime();
}

template <typename Container, typename Comparator>
int DijkstraBalancer<Container, Comparator>::getDelta(int curTime, 
    const model::Chan* curChan) {
  // Compute delta between neighbouring nodes
  if (mode == LatencyBalanceMode::ASAP) {
    return curTime - (nodeMap[curChan->source.node] + (int)curChan->ind.ticks);
  }
  if (mode == LatencyBalanceMode::ALAP) {
    return (nodeMap[curChan->source.node] - (int)curChan->ind.ticks) - curTime;
  }
  return -1;
};
