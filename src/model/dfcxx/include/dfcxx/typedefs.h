//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_TYPEDEFS_H
#define DFCXX_TYPEDEFS_H

#include <unordered_map>

namespace dfcxx {

enum Ops {
  // Arithmetic operations.
  ADD_INT = 1,
  ADD_FLOAT,
  SUB_INT,
  SUB_FLOAT,
  MUL_INT,
  MUL_FLOAT,
  DIV_INT,
  DIV_FLOAT,
  NEG_INT,
  NEG_FLOAT,
  // Bitwise operations.
  AND_INT,
  AND_FLOAT,
  OR_INT,
  OR_FLOAT,
  XOR_INT,
  XOR_FLOAT,
  NOT_INT,
  NOT_FLOAT,
  // Comparison operations.
  LESS_INT,
  LESS_FLOAT,
  LESSEQ_INT,
  LESSEQ_FLOAT,
  GREATER_INT,
  GREATER_FLOAT,
  GREATEREQ_INT,
  GREATEREQ_FLOAT,
  EQ_INT,
  EQ_FLOAT,
  NEQ_INT,
  NEQ_FLOAT,
  // Utility values. Contain normal and incremented number of elements in the enum.
  INC_COUNT,
  COUNT = INC_COUNT - 1
};

enum Scheduler {
  Linear = 0,
  ASAP
};

} // namespace dfcxx

typedef std::unordered_map<dfcxx::Ops, unsigned> DFLatencyConfig;

#endif // DFCXX_TYPEDEFS_H
