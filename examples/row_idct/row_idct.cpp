//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "row_idct.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  RowIDCT *kernel = new RowIDCT();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
