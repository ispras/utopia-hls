//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/parser/dfc/stream.h"

namespace dfc {

void wire::declare(const wire *var) {
  // TODO:
}

void wire::connect(const wire *in, const wire *out) {
  // TODO:
}

void wire::connect(const std::string &op,
                   const std::vector<const wire*> &in,
                   const std::vector<const wire*> &out) {
  // TODO:
}

} // namespace dfc
