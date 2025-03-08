//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_UTILS_H
#define DFCXX_UTILS_H

#include "dfcxx/graph.h"

#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dfcxx {

std::vector<Node *> topSort(const Graph &graph);

} // namespace dfcxx

#endif // DFCXX_UTILS_H
