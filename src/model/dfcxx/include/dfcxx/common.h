//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_COMMON_H
#define DFCXX_COMMON_H

#include <cstdint>
#include <string>

namespace dfcxx {

struct ModuleParam {
  enum class Kind {
    INT,
    UINT,
    REAL
  };

  std::string name;
  Kind kind;
  union {
    int64_t int_;
    uint64_t uint_;
    double real_;
  } value;
};

} // namespace dfcxx

#endif // DFCXX_COMMON_H
