//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CHANNEL_H
#define DFCXX_CHANNEL_H

#include "dfcxx/node.h"

namespace dfcxx {

struct Channel {
  Node *source;
  Node *target;
  unsigned opInd;

  Channel() = default;
  Channel(Node *source, Node *target, unsigned opInd);

  bool operator==(const Channel &channel) const;
};

} // namespace dfcxx

template <>
struct std::hash<dfcxx::Channel> {
  size_t operator()(const dfcxx::Channel &ch) const noexcept {
    return std::hash<dfcxx::DFVariableImpl *>()(ch.source->var) *
           std::hash<dfcxx::DFVariableImpl *>()(ch.target->var) +
           ch.opInd;
  }
};

#endif // DFCXX_CHANNEL_H
