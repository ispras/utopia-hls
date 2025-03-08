//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CONSTANT_H
#define DFCXX_CONSTANT_H

#include "dfcxx/kernmeta.h"

namespace dfcxx {

class Kernel;

class Constant {
  friend Kernel;

private:
  KernMeta &meta;

  Constant(KernMeta &meta);

public:
  DFVariable var(const DFType &type, int64_t value);

  DFVariable var(const DFType &type, uint64_t value);

  DFVariable var(const DFType &type, double value);
};

} // namespace dfcxx

#endif // DFCXX_CONSTANT_H
