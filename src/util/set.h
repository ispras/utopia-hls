//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <set>

namespace eda::utils {

template <typename C, typename P>
void discard_if(C &c, P pred) {
  auto i = c.begin();
  auto end = c.end();

  while (i != end) {
    if (pred(*i)) {
      i = c.erase(i);
    } else {
      i++;
    }
  }
}

} // namespace eda::utils
