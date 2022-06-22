//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <string>

#include "util/path.h"

namespace eda::utils {

std::string correctPath(const std::string& path) {
  return path + (path.back() != '/' ? "/" : "");
}

} // namespace eda::utils
