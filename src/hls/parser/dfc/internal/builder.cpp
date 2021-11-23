//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/parser/dfc/internal/builder.h"

using namespace eda::hls::model;

namespace eda::hls::parser::dfc {

std::shared_ptr<Model> Builder::create(const std::string &name) {
  auto *model = new Model(name);
  // TODO:
  return std::shared_ptr<Model>(model);
}

} // namespace eda::hls::parser::dfc
