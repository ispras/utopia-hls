//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CONSTANT_H
#define DFCXX_CONSTANT_H

#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/varbuilders/builder.h"

namespace dfcxx {

class Kernel;

class Constant {
  friend Kernel;

private:
  Graph &graph;
  GraphHelper helper;
  VarBuilder &varBuilder;
  KernStorage &storage;

  Constant(Graph &graph, TypeBuilder &typeBuilder,
           VarBuilder &varBuilder, KernStorage &storage);

public:
  DFVariable var(const DFType &type, int64_t value);

  DFVariable var(const DFType &type, uint64_t value);

  DFVariable var(const DFType &type, double value);
};

} // namespace dfcxx

#endif // DFCXX_CONSTANT_H
