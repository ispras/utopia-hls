//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CONTROL_H
#define DFCXX_CONTROL_H

#include "dfcxx/graph.h"
#include "dfcxx/kernstorage.h"
#include "dfcxx/varbuilders/builder.h"

#include <initializer_list>

namespace dfcxx {

class Kernel;

class Control {
  friend Kernel;
private:
  Graph &graph;
  GraphHelper helper;
  VarBuilder &varBuilder;
  KernStorage &storage;

  Control(Graph &graph, TypeBuilder &typeBuilder, VarBuilder &varBuilder,
          KernStorage &storage);

public:
  DFVariable mux(DFVariable ctrl, std::initializer_list<DFVariable> args);
};

} // namespace dfcxx

#endif // DFCXX_CONTROL_H
