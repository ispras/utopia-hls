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
// It is assumed that graph G(V,E) implements the following methods:
//
// std::size_t  G::nNodes()         const;
// std::size_t  G::nEdges()         const;
// container<V> G::getSources()     const;
// container<E> G::getOutEdges(V v) const;
// V            G::leadsTo(E e)     const;
//
//===----------------------------------------------------------------------===//

/// Performs topological sorting of the directed graph (feedbacks are ignored).
template <typename G, typename V, typename E>
std::vector<V> topologicalSort(const G &graph) {
  // DFS stack.
  std::stack<std::pair<V, std::size_t>> searchStack;

  // Set of visited nodes.
  std::unordered_set<V> visitedNodes;
  visitedNodes.reserve(graph.nNodes());

  // Set of nodes in topological order.
  std::vector<V> sortedNodes(graph.nNodes());
  auto it = sortedNodes.rbegin();

  // Collect the source nodes for the DFS traveral.
  for (auto v : graph.getSources()) {
    searchStack.push({v, 0});
    visitedNodes.insert(v);
  }

  // DFS traveral.
  while (!searchStack.empty()) {
    auto &[v, i] = searchStack.top();
    const auto &out = graph.getOutEdges(v);

    // Schedule the next node.
    bool hasMoved = false;

    while (i < out.size()) {
      auto e = out[i++];
      auto n = graph.leadsTo(e);

      // The next node is unvisited.
      if (visitedNodes.find(n) == visitedNodes.end()) {
        searchStack.push({n, 0});
        visitedNodes.insert(n);

        hasMoved = true;
        break;
      }
    }

    // All successors of the node has been traversed.
    if (!hasMoved) {
      *(it++) = v;
      searchStack.pop();
    }
  }

  assert(visitedNodes.size() == graph.nNodes());
  return sortedNodes;
}

/// Traverses the graph in topological order and handles the nodes.
template <typename G, typename V, typename E>
void traverseTopologicalOrder(const G &graph,
                              std::function<void(V)> handleNode) {
  std::vector<V> sortedNodes = topologicalSort<G, V, E>(graph);
  for (auto v : sortedNodes) {
    handleNode(v);
  }
}

/// Traverses the graph in topological order and handles the nodes and edges.
template <typename G, typename V, typename E>
void traverseTopologicalOrder(const G &graph,
                              std::function<void(V)> handleNode,
                              std::function<void(E)> handleEdge) {
  std::vector<V> sortedNodes = topologicalSort<G, V, E>(graph);
  for (auto v : sortedNodes) {
    handleNode(v);
    for (auto e : graph.getOutEdges(v)) {
      handleEdge(e);
    }
  }
}

} // namespace eda::utils::graph
