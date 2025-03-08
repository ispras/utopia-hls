//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "scalar3.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  Scalar3 *kernel = new Scalar3();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
