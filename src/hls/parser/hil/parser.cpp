//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "hls/parser/hil/parser.h"

// The parser is built w/ the prefix 'hh' (not 'yy').
extern FILE *hhin;
extern int hhparse(void);

namespace eda::hls::parser::hil {

int parse(const std::string &filename) {
  FILE *file = fopen(filename.c_str(), "r");
  if (file == nullptr) {
    return -1;
  }

  hhin = file;
  return hhparse();
}

} // namespace eda::hls::parser::hil
