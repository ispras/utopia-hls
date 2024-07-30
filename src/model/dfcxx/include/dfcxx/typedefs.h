//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2024 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_TYPEDEFS_H
#define DFCXX_TYPEDEFS_H

#include <cstdint>
#include <string>
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

// Used for accessing specified output format paths.
enum class OutputFormatID : uint8_t {
  SystemVerilog = 0,
  DFCIR,
  // Utility value. Constains the number of elements in the enum.
  COUNT
};

#define OUT_FORMAT_ID_INT(id) static_cast<uint8_t>(dfcxx::OutputFormatID::id)

} // namespace dfcxx

typedef std::unordered_map<dfcxx::Ops, unsigned> DFLatencyConfig;
typedef std::unordered_map<dfcxx::OutputFormatID, std::string> DFOutputPaths;

#endif // DFCXX_TYPEDEFS_H
