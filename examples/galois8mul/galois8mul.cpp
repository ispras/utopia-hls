//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "galois8mul.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  Galois8Mul *kernel = new Galois8Mul();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
