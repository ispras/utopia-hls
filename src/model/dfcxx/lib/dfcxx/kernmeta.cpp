//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernmeta.h"

namespace dfcxx {

void KernMeta::transferFrom(KernMeta &&meta) {
  graph.transferFrom(std::move(meta.graph));
  storage.transferFrom(std::move(meta.storage));
}

} // namespace dfcxx
