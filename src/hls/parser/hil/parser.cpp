//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
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

std::unique_ptr<eda::hls::model::Model> parse(const std::string &filename) {
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

} // namespace eda::hls::parser::hil
