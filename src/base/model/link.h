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

//===----------------------------------------------------------------------===//
// Link
//===----------------------------------------------------------------------===//

/**
 * \brief Represents a connection between two nodes.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
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
    if (&rhs == this) {
      return true;
    }

    return source == rhs.source
        && target == rhs.target
        && input  == rhs.input;
  }

  /// Source node.
  N source;
  /// Target node.
  N target;
  /// Target input index.
  size_t input;
};

} // namespace eda::base::model

//===----------------------------------------------------------------------===//
// Hash
//===----------------------------------------------------------------------===//

namespace std {

/**
 * \brief Implements hash code computation for links.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
template <typename N>
struct hash<eda::base::model::Link<N>> {
  size_t operator()(const eda::base::model::Link<N> &link) const {
    const size_t prime = 37;

    size_t hash = std::hash<N>()(link.source);
    hash *= prime;
    hash += std::hash<N>()(link.target);
    hash *= prime;
    hash += link.input;

    return hash;
  }
};

} // namespace std
