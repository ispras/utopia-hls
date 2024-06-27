//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "movingsum.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  MovingSum *kernel = new MovingSum();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
