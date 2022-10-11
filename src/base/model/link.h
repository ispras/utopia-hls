//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace eda::base::model {

/// Represents a connection between two nodes.
template<typename N>
struct Link final {
  using List = std::vector<Link<N>>;

  // General link.
  Link(N source, N target, std::size_t input):
    source(source), target(target), input(input) {}

  // Self-link (a port).
  explicit Link(N node): Link(node, node, 0) {}

  bool isPort() const {
    return source == target;
  }

  bool operator ==(const Link<N> &rhs) const {
    return source == rhs.source && target == rhs.target && input == rhs.input;
  }

  /// Source node.
  N source;
  /// Target node.
  N target;
  /// Target input.
  std::size_t input;
};

} // namespace eda::base::model

//===----------------------------------------------------------------------===//
// Hash
//===----------------------------------------------------------------------===//

namespace std {

/// Hash for the Link class.
template <typename N>
struct hash<eda::base::model::Link<N>> {
  std::size_t operator()(const eda::base::model::Link<N> &link) const {
    std::size_t h = std::hash<N>()(link.source);
    h *= 37;
    h += std::hash<N>()(link.target);
    h *= 37;
    h += link.input;
    return h;
  }
};

} // namespace std
