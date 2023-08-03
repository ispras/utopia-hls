//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
//
// HIL-to-mlir printer.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include <string>

using Model = eda::hls::model::Model;

namespace eda::hls::model {

std::ostream &dumpModelMlir(const Model &model, std::ostream &os);
void dumpModelMlirToFile(const Model &model, const std::string &filename);

} // namespace eda::hls::model