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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace eda::utils::graph {

/// Traverses the graph in topological order.
/// 
/// The following methods are assumed: 
///
/// const std::vector<V> &Graph::getNodes() const
/// const std::vector<V> &Graph::getSources() const
/// const std::vector<E> &Graph::getOutEdges(V v) const
/// V                     Graph::getSource(E e) const
/// V                     Graph::getTarget(E e) const
template <typename G, typename V, typename E>
void traverseTopologicalOrder(G &graph,
                              std::function<void(V)> handleNode,
                              std::function<void(E)> handleEdge) {
  // DFS stack.
  std::stack<std::pair<V, std::size_t>> stack;
  // Set of visited nodes.
  std::unordered_set<V> visited;

  // Collect the source nodes for the DFS traveral.
  for (auto v : graph.getSources()) {
    stack.push({v, 0});
    visited.insert(v);
  }

  // Traverse the nodes in topological order (DFS).
  while (!stack.empty()) {
    auto &[v, i] = stack.top();
    const auto &edges = graph.getOutEdges(v);

    // Visit the node once (when added to the stack).
    if (i == 0) {
      handleNode(v);

      for (auto e : edges) {
        handleEdge(e);
      }
    }

    // Schedule the next node (implicit topological sort).
    bool hasMoved = false;

    while (i < edges.size()) {
      auto e = edges[i++];
      auto n = graph.getTarget(e);

      // The next node is unvisited.
      if (visited.find(n) == visited.end()) {
        stack.push({n, 0});
        visited.insert(n);
        hasMoved = true;
        break;
      }
    }

    // All successors of the node has been traversed.
    if (!hasMoved) {
      stack.pop();
    }
  }
}

} // namespace eda::utils::graph
