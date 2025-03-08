//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/utils.h"

#include <cassert>

namespace dfcxx {

std::vector<Node *> topSort(const Graph &graph) {
  size_t nodesCount = graph.getNodes().size();
  auto &outs = graph.getOutputs();

  std::vector<Node *> result(nodesCount);

  std::unordered_map<Node *, size_t> checked;
  std::stack<Node *> stack;

  for (Node *node: graph.getStartNodes()) {
    stack.push(node);
    checked[node] = 0;
  }

  size_t i = nodesCount;
  while (!stack.empty() && i > 0) {
    Node *node = stack.top();
    size_t count = outs.at(node).size();
    size_t curr;
    bool flag = true;
    for (curr = checked[node]; flag && curr < count; ++curr) {
      Channel *next = outs.at(node)[curr];
      if (checked.find(next->target) == checked.end()) {
        checked[next->target] = 0;
        stack.push(next->target);
        flag = false;
      }
      ++checked[node];
    }

    if (flag) {
      stack.pop();
      result[--i] = node;
    }
  }
  assert(stack.empty());
  assert(i == 0);
  return result;
}

} // namespace dfcxx
