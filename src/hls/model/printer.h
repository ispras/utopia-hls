//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ostream>

#include "hls/model/model.h"

namespace eda::hls::model {

void printDot(std::ostream &out, const Model &model);

} // namespace eda::hls::model