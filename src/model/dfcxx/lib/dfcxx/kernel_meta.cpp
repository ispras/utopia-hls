//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/kernel_meta.h"

namespace dfcxx {

KernelMeta *KernelMeta::top = nullptr;

KernelMeta::KernelMeta() {
  if (KernelMeta::top == nullptr) {
    KernelMeta::top = this;
  }
}

KernelMeta::~KernelMeta() {
  if (KernelMeta::top == this) {
    KernelMeta::top = nullptr;
  }
}

void KernelMeta::transferFrom(KernelMeta &&meta) {
  meta.graph.resetMeta(this);
  graph.transferFrom(std::move(meta.graph));
  storage.transferFrom(std::move(meta.storage));
}

} // namespace dfcxx
