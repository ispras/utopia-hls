//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_KERNEL_META_H
#define DFCXX_KERNEL_META_H

#include "dfcxx/graph.h"
#include "dfcxx/kernel_storage.h"
#include "dfcxx/type_builder.h"
#include "dfcxx/var_builder.h"

namespace dfcxx {

struct KernelMeta {
  static KernelMeta *top;

  Graph graph;
  KernelStorage storage;
  TypeBuilder typeBuilder;
  VarBuilder varBuilder;

  KernelMeta() = default;
  KernelMeta(const KernelMeta &) = delete;
  ~KernelMeta() = default;

  void transferFrom(KernelMeta &&meta);
};

} // namespace dfcxx

#endif // DFCXX_KERNEL_META_H
