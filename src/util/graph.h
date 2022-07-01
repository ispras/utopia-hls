//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <stack>
#include <unordered_set>
#include <vector>

namespace eda::utils::graph {

//===----------------------------------------------------------------------===//
//
// It is assumed that graph G contains the following types and methods:
//
// G::V (node type);
// G::E (edge type);
//
// std::size_t  G::nNodes()         const;
// std::size_t  G::nEdges()         const;
// container<V> G::getSources()     const;
// container<E> G::getOutEdges(V v) const;
// V            G::leadsTo(E e)     const.
//
//===----------------------------------------------------------------------===//

/// Performs topological sorting of the directed graph (feedbacks are ignored).
template <typename G, typename OutEdgeContainer = std::vector<typename G::E>>
std::vector<typename G::V> topologicalSort(const G &graph) {
  using V = typename G::V;
  using OutEdgeIterator = typename OutEdgeContainer::const_iterator;

  std::vector<V> sortedNodes(graph.nNodes());
  auto it = sortedNodes.rbegin();

  std::stack<std::pair<V, OutEdgeIterator>> searchStack;

  std::unordered_set<V> visitedNodes;
  visitedNodes.reserve(graph.nNodes());

  // Collect the source nodes for the DFS traveral.
  for (auto v : graph.getSources()) {
    const OutEdgeContainer &out = graph.getOutEdges(v);

    searchStack.push({v, out.cbegin()});
    visitedNodes.insert(v);
  }

  // DFS traveral.
  while (!searchStack.empty()) {
    auto &[v, i] = searchStack.top();
    const OutEdgeContainer &out = graph.getOutEdges(v);

    // Schedule the next node.
    bool hasMoved = false;

    while (i != out.cend()) {
      auto n = graph.leadsTo(*i++);

      // The next node is unvisited.
      if (visitedNodes.find(n) == visitedNodes.end()) {
        const OutEdgeContainer &out = graph.getOutEdges(n);

        searchStack.push({n, out.cbegin()});
        visitedNodes.insert(n);

        hasMoved = true;
        break;
      }
    }

    // All successors of the node has been traversed.
    if (!hasMoved) {
      *it++ = v;
      searchStack.pop();
    }
  }

  assert(visitedNodes.size() == graph.nNodes());
  return sortedNodes;
}

/// Traverses the graph in topological order and handles the nodes.
template <typename G, typename OutEdgeContainer = std::vector<typename G::E>>
void traverseTopologicalOrder(const G &graph,
                              std::function<void(typename G::V)> handleNode) {
  std::vector<typename G::V> sortedNodes =
    topologicalSort<G, OutEdgeContainer>(graph);
  for (auto v : sortedNodes) {
    handleNode(v);
  }
}

/// Traverses the graph in topological order and handles the nodes and edges.
template <typename G, typename OutEdgeContainer = std::vector<typename G::E>>
void traverseTopologicalOrder(const G &graph,
                              std::function<void(typename G::V)> handleNode,
                              std::function<void(typename G::E)> handleEdge) {
  std::vector<typename G::V> sortedNodes =
    topologicalSort<G, OutEdgeContainer>(graph);
  for (auto v : sortedNodes) {
    handleNode(v);
    for (auto e : graph.getOutEdges(v)) {
      handleEdge(e);
    }
  }
}

} // namespace eda::utils::graph
