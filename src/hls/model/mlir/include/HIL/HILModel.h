//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"

#include <string>

#include "hls/model/model.h"

namespace eda::hls::model {
std::unique_ptr<eda::hls::model::Model>
parse_model_from_mlir(const std::string &s);
std::unique_ptr<eda::hls::model::Model>
parse_model_from_mlir_file(const std::string &filename);
std::string dump_model_to_mlir(eda::hls::model::Model &m);
} // namespace eda::hls::model
