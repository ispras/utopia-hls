//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "muxmul.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  MuxMul *kernel = new MuxMul();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
