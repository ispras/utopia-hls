//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>

#include "HIL/Model.h"

namespace eda::hls::model {
  struct Model;
} // namespace eda::hls::model

namespace eda::hls::parser::hil {

std::shared_ptr<eda::hls::model::Model> parse(const std::string &filename);
std::shared_ptr<mlir::hil::Model> parseToMlir(const std::string &filename);

} // namespace eda::hls::parser::hil
