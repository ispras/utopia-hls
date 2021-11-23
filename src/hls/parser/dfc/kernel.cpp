//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/parser/dfc/kernel.h"
#include "hls/parser/dfc/internal/builder.h"

using namespace eda::hls::parser::dfc;

namespace dfc {

kernel::kernel(const std::string &name, const params &args): name(name) {
  Builder::get().startKernel(name);
}

} // namespace dfc
