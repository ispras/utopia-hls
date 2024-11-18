//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "polynomial2_inst.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  Polynomial2Inst *kernel = new Polynomial2Inst();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
