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
#include "dfcxx/typedefs.h"

#include <string>
#include <vector>

namespace dfcxx {

std::vector<std::string> outputPathsToVector(const DFOutputPaths &outputPaths);
std::vector<Node *> topSort(const Graph &graph);

} // namespace dfcxx

#endif // DFCXX_UTILS_H
