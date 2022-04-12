//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "hls/model/model.h"
#include "hls/parser/hil/builder.h"
#include "hls/parser/hil/parser.h"

// The parser is built w/ the prefix 'hh' (not 'yy').
extern FILE *hhin;
extern int hhparse(void);

namespace eda::hls::parser::hil {

std::shared_ptr<eda::hls::model::Model> parse(const std::string &filename) {
  FILE *file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
    return nullptr;
  }

  hhin = file;
  if (hhparse() == -1) {
    return nullptr;
  }

  return Builder::get().create();
}

std::shared_ptr<mlir::hil::Model> parseToMlir(const std::string &filename) {

  std::shared_ptr<Model> hilModel = parse(filename);
  mlir::hil::Model mlirModel =
      mlir::model::MLIRModule::load_from_model(*hilModel.get()).get_root();

  return std::unique_ptr<mlir::hil::Model>(&mlirModel);
}

} // namespace eda::hls::parser::hil
