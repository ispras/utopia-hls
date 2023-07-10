//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <string>

#include "utils/string.h"

namespace eda::utils {

std::string unique_name(const std::string &prefix) {
  static int i = 0;
  return utils::format("%s_%d", prefix.c_str(), i++);
}

} // namespace eda::utils