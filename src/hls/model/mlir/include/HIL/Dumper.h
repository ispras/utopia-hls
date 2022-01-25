//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "hls/model/model.h"

namespace eda::hls::model {
std::ostream &dump_model_mlir(const eda::hls::model::Model &model,
                              std::ostream &os);
void dump_model_mlir_to_file(const eda::hls::model::Model &model,
                             const std::string &filename);
} // namespace eda::hls::model
