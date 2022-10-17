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

namespace eda::base::model {

template <typename F, typename N>
struct NodeHashKey final {
  using Signal = eda::base::model::Signal<N>;
  using SignalList = typename Signal::List;
 
  NodeHashKey(F func, const SignalList &inputs):
      func(static_cast<uint16_t>(func)),
      arity(static_cast<uint16_t>(inputs.size())) {
    ihash = 0;
    for (const auto &input : inputs) {
      ihash *= 37;
      ihash += input.node();
    }
  }

  constexpr bool operator ==(const NodeHashKey &rhs) const {
    return func == rhs.func
        && arity == rhs.arity
        && ihash == rhs.ihash;
  }

  uint16_t func;
  uint16_t arity;
  uint64_t ihash;
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
    hash += key.ihash;

    return hash;
  }
};

} // namespace std
