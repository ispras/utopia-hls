//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/parser/dfc/stream.h"

namespace dfc {

void stream::declare(const stream *var) {
  // TODO:
}

void stream::connect(const stream *in, const stream *out) {
  // TODO:
}

void stream::connect(const std::string &op,
                     const std::vector<const stream*> &in,
                     const std::vector<const stream*> &out) {
  // TODO:
}

} // namespace dfc
