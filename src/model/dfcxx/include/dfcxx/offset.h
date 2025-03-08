//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_OFFSET_H
#define DFCXX_OFFSET_H

#include "dfcxx/kernel_meta.h"

namespace dfcxx {

class Kernel;

class Offset {
  friend Kernel;

private:
  KernelMeta &meta;

  Offset(KernelMeta &meta);

public:
  DFVariable operator()(DFVariable &stream, int64_t offset);
};

} // namespace dfcxx

#endif // DFCXX_OFFSET_H
