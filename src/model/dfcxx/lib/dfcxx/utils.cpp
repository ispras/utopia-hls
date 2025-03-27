//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/utils.h"

#include <cassert>
#include <stack>
#include <unordered_map>

namespace dfcxx {

std::vector<std::string> outputPathsToVector(const DFOutputPaths &outputPaths) {
  std::vector<std::string> result(OUT_FORMAT_ID_INT(COUNT), "");
  for (const auto &[id, path] : outputPaths) {
    result[static_cast<uint8_t>(id)] = path;
  }
  return result;
}

std::vector<Node *> topSort(const Graph &graph) {
  size_t nodesCount = graph.getNodes().size();

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
    size_t count = node->outputs.size();
    size_t curr;
    bool flag = true;
    for (curr = checked[node]; flag && curr < count; ++curr) {
      Channel *next = node->outputs[curr];
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
