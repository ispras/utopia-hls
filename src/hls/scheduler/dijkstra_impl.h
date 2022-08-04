//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

using StdChanQueue = std::priority_queue<const Chan*, std::vector<const Chan*>, CompareChan>;
using GenericChanQueue = Queue<const Chan*, std::priority_queue, std::vector<const Chan*>, CompareChan>;

template<>
inline GenericChanQueue::Queue(void *comparator) {
  container = new StdChanQueue(*(static_cast<CompareChan*>(comparator)));
}

template<>
inline const Chan* GenericChanQueue::front() {
    return container->top();
}

template<>
inline const Chan* Queue<const Chan*, std::priority_queue>::front() {
    return container->top();
}

template <template <typename, typename...> class C, typename... Ts>
void DijkstraBalancer<C, Ts...>::initQueue() {
  delete toVisit;
  toVisit = new Queue<const Chan*, C, Ts...>();
}

template <>
inline void DijkstraBalancer<std::priority_queue, std::vector<const Chan*>, CompareChan>::initQueue() {
  delete toVisit;
  toVisit = new GenericChanQueue(new CompareChan(nodeMap));
}

template <template <typename, typename...> class C, typename... Ts>
void DijkstraBalancer<C, Ts...>::init(const Graph *graph) {
  TraverseBalancerBase::init(*graph);
  initQueue();
  currentNode = nullptr;
}

template <template <typename, typename...> class C, typename... Ts>
void DijkstraBalancer<C, Ts...>::visitChan(const Chan *chan) {
  const Node *targetNode = getNextNode(chan);
  // Update destination node time
  if (targetNode && currentNode) {
    nodeMap[targetNode] = std::max(nodeMap[currentNode] + (int)chan->ind.ticks, 
      nodeMap[targetNode]);
    graphTime = std::max((unsigned)nodeMap[targetNode], graphTime);
  }
}

template <template <typename, typename...> class C, typename... Ts>
void DijkstraBalancer<C, Ts...>::visitNode(const Node *node) {
  currentNode = node;
  for (const auto *chan : getConnections(node)) {
    toVisit->push(chan);
    visitChan(chan);
  }
}

template <template <typename, typename...> class C, typename... Ts>
void DijkstraBalancer<C, Ts...>::start(const std::vector<Node*> &startNodes) {
  for (const auto *node : startNodes) {
    visitNode(node);
  }
}

template <template <typename, typename...> class C, typename... Ts>
const Node* DijkstraBalancer<C, Ts...>::getNextNode(const Chan *chan) {
  return 
    mode == LatencyBalanceMode::ASAP ? chan->target.node : chan->source.node;
}

template <template <typename, typename...> class C, typename... Ts>
const std::vector<Chan*>& DijkstraBalancer<C, Ts...>::getConnections(const Node *next) {
  return mode == LatencyBalanceMode::ASAP ? next->outputs : next->inputs;
}

template <template <typename, typename...> class C, typename... Ts>
void DijkstraBalancer<C, Ts...>::traverse(const std::vector<Node*> &startNodes) {
  start(startNodes);
  while (!toVisit->empty()) {
    visitNode(getNextNode(toVisit->front()));
    toVisit->pop();
  }
}

template <template <typename, typename...> class C, typename... Ts>
void DijkstraBalancer<C, Ts...>::balance(Model &model, LatencyBalanceMode balanceMode) {
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

template <template <typename, typename...> class C, typename... Ts>
int DijkstraBalancer<C, Ts...>::getDelta(int curTime, const Chan* curChan) {
  // Compute delta between neighbouring nodes
  if (mode == LatencyBalanceMode::ASAP) {
    return curTime - (nodeMap[curChan->source.node] + (int)curChan->ind.ticks);
  }
  if (mode == LatencyBalanceMode::ALAP) {
    return (nodeMap[curChan->source.node] - (int)curChan->ind.ticks) - curTime;
  }
  return -1;
};
