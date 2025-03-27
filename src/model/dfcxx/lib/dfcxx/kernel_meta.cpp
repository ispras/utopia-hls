//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernel_meta.h"

namespace dfcxx {

void KernelMeta::transferFrom(KernelMeta &&meta) {
  meta.graph.resetMeta(this);
  graph.transferFrom(std::move(meta.graph));
  storage.transferFrom(std::move(meta.storage));
}

} // namespace dfcxx
