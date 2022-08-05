//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

template<>
inline GenericChanQueue::Queue(void *comparator) {
  container = new StdPriorityQueue(*(static_cast<CompareChan*>(comparator)));
}

template<>
inline const Chan* GenericChanQueue::front() {
    return container->top();
}

template <typename C>
void DijkstraBalancer<C>::initQueue() {
  delete toVisit;
  toVisit = new Queue<const Chan*, C>();
}

template <>
inline void DijkstraBalancer<StdPriorityQueue>::initQueue() {
  delete toVisit;
  toVisit = new GenericChanQueue(new CompareChan(nodeMap));
}

template <typename C>
void DijkstraBalancer<C>::init(const Graph *graph) {
  TraverseBalancerBase::init(*graph);
  initQueue();
  currentNode = nullptr;
}

template <typename C>
void DijkstraBalancer<C>::visitChan(const Chan *chan) {
  const Node *targetNode = getNextNode(chan);
  // Update destination node time
  if (targetNode && currentNode) {
    nodeMap[targetNode] = std::max(nodeMap[currentNode] + (int)chan->ind.ticks, 
      nodeMap[targetNode]);
    graphTime = std::max((unsigned)nodeMap[targetNode], graphTime);
  }
}

template <typename C>
void DijkstraBalancer<C>::visitNode(const Node *node) {
  currentNode = node;
  for (const auto *chan : getConnections(node)) {
    toVisit->push(chan);
    visitChan(chan);
  }
}

template <typename C>
void DijkstraBalancer<C>::start(const std::vector<Node*> &startNodes) {
  for (const auto *node : startNodes) {
    visitNode(node);
  }
}

template <typename C>
const Node* DijkstraBalancer<C>::getNextNode(const Chan *chan) {
  return 
    mode == LatencyBalanceMode::ASAP ? chan->target.node : chan->source.node;
}

template <typename C>
const std::vector<Chan*>& DijkstraBalancer<C>::getConnections(const Node *next) {
  return mode == LatencyBalanceMode::ASAP ? next->outputs : next->inputs;
}

template <typename C>
void DijkstraBalancer<C>::traverse(const std::vector<Node*> &startNodes) {
  start(startNodes);
  while (!toVisit->empty()) {
    const Node *next = getNextNode(toVisit->front());
    toVisit->pop();
    visitNode(next);
  }
}

template <typename C>
void DijkstraBalancer<C>::balance(Model &model, LatencyBalanceMode balanceMode) {
  mode = balanceMode;
  const Graph *graph = model.main();
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

template <typename C>
int DijkstraBalancer<C>::getDelta(int curTime, const Chan* curChan) {
  // Compute delta between neighbouring nodes
  if (mode == LatencyBalanceMode::ASAP) {
    return curTime - (nodeMap[curChan->source.node] + (int)curChan->ind.ticks);
  }
  if (mode == LatencyBalanceMode::ALAP) {
    return (nodeMap[curChan->source.node] - (int)curChan->ind.ticks) - curTime;
  }
  return -1;
};
