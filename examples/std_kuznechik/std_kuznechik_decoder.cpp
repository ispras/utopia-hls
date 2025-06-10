//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "dfcxx/std/crypto/gost_34_12/kuznechik.h"

#include <memory>

std::unique_ptr<dfcxx::Kernel> start() {
  using dfcxx::std::KuznechikDecoder;
  KuznechikDecoder *kernel = new KuznechikDecoder(true);
  return std::unique_ptr<dfcxx::Kernel>(kernel);
}
