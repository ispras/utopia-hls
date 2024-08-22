//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_KERNMETA_H
#define DFCXX_KERNMETA_H

#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/typebuilders/builder.h"
#include "dfcxx/varbuilders/builder.h"

namespace dfcxx {

struct KernelMeta {
  Graph graph;
  KernStorage storage;
  TypeBuilder typeBuilder;
  VarBuilder varBuilder;

  KernelMeta() = default;
  KernelMeta(const KernelMeta &) = delete;
  ~KernelMeta() = default;
};

} // namespace dfcxx

#endif // DFCXX_KERNMETA_H
