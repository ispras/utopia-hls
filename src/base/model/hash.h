//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "base/model/signal.h"

#include <algorithm>
#include <cstdint>

namespace eda::base::model {

/**
 * \brief Represents an inexact (loosy) key for hashing net nodes.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
template <typename F, typename N>
struct NodeHashKey final {
  using Signal = eda::base::model::Signal<N>;
  using SignalList = typename Signal::List;

  /// Constructs a key from the given node signature.
  NodeHashKey(F func, const SignalList &inputs):
      func(static_cast<uint16_t>(func)),
      arity(static_cast<uint16_t>(inputs.size())) {
    const uint64_t prime = 37;

    std::vector<size_t> hashes(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) {
      hashes[i] = std::hash<N>()(inputs[i].node());
    }

    if (func.isCommutative()) {
      std::sort(hashes.begin(), hashes.end());
    }

    hash2 = hash1 = 0;
    for (const auto hash : hashes) {
      hash1 ^= hash;
      hash2 *= prime;
      hash2 += hash;
    }
  }

  /// Compares this key w/ the given one.
  constexpr bool operator ==(const NodeHashKey &rhs) const {
    return func == rhs.func
        && arity == rhs.arity
        && hash1 == rhs.hash1
        && hash2 == rhs.hash2;
  }

  /// Functional symbol (exact).
  uint16_t func;
  /// Node arity (exact).
  uint16_t arity;
  /// Hash code(s) of node inputs.
  uint32_t hash1;
  uint64_t hash2;
};

} // namespace eda::base::model

//===----------------------------------------------------------------------===//
// Hash
//===----------------------------------------------------------------------===//

namespace std {

/**
 * \brief Implements hash code computation for NodeHashKey.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
template <typename F, typename N>
struct hash<eda::base::model::NodeHashKey<F, N>> {
  size_t operator()(const eda::base::model::NodeHashKey<F, N> &key) const {
    const size_t prime = 37;

    size_t hash = key.func;
    hash *= prime;
    hash += key.arity;
    hash *= prime;
    hash += key.hash1;
    hash *= prime;
    hash += key.hash2;

    return hash;
  }
};

} // namespace std
