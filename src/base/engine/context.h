//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "base/engine/storage.h"

namespace eda::base::engine {

struct Context final {
  Context(unsigned runId, Storage &store):
      runId(runId), store(store) {}

  const unsigned runId;
  Storage &store;
};

} // namespace eda::base::engine
