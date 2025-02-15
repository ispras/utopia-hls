//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "kuznechik_encoder.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  KuznechikEncoder *kernel = new KuznechikEncoder();
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
