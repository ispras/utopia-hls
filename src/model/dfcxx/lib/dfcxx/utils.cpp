//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/utils.h"

namespace dfcxx {

std::vector<Node> topSort(std::unordered_set<Node> startNodes,
                          std::unordered_map<Node,
                                             std::vector<Channel>> outs,
                          size_t nodesCount) {
  std::vector<Node> result(nodesCount);

  std::unordered_map<Node, size_t> checked;
  std::stack<Node> stack;

  for (Node node: startNodes) {
    stack.push(node);
    checked[node] = 0;
  }

  size_t i = nodesCount - 1;
  while (!stack.empty()) {
    Node node = stack.top();
    size_t count = outs[node].size();
    size_t curr;
    bool flag = true;
    for (curr = checked[node]; flag && curr < count; ++curr) {
      Channel next = outs[node][curr];
      if (!checked[next.target]) {
        stack.push(next.target);
        flag = false;
      }
      ++checked[node];
    }

    if (flag) {
      stack.pop();
      result[i--] = node;
    }
  }
  return result;
}

} // namespace dfcxx
