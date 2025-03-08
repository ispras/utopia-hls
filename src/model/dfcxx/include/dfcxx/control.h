//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_CONTROL_H
#define DFCXX_CONTROL_H

#include "dfcxx/kernmeta.h"

#include <initializer_list>

namespace dfcxx {

class Kernel;

class Control {
  friend Kernel;

private:
  KernMeta &meta;

  Control(KernMeta &meta);

public:
  DFVariable mux(DFVariable ctrl, std::initializer_list<DFVariable> args);
};

} // namespace dfcxx

#endif // DFCXX_CONTROL_H
