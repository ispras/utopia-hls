//===----------------------------------------------------------------------===//
//
// Part of the Utopia HLS Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2024-2025 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#ifndef DFCXX_TYPEDEFS_H
#define DFCXX_TYPEDEFS_H

#include <cstdint>
#include <initializer_list>
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
  ASAP,
  CombPipelining
};

// Used for accessing specified output format paths.
enum class OutputFormatID : uint8_t {
  SystemVerilog = 0,
  SVLibrary,
  UnscheduledDFCIR,
  ScheduledDFCIR,
  FIRRTL,
  DOT,
  // Utility value. Constains the number of elements in the enum.
  COUNT
};

#define OUT_FORMAT_ID_INT(id) static_cast<uint8_t>(dfcxx::OutputFormatID::id)

struct DFLatencyConfig {
public:
  std::unordered_map<Ops, uint16_t> internalOps;
  std::unordered_map<std::string, uint16_t> externalOps;

  DFLatencyConfig() = default;

  DFLatencyConfig(const DFLatencyConfig &) = default;

  DFLatencyConfig(
      std::initializer_list<std::pair<const Ops, uint16_t>> internals,
      std::initializer_list<std::pair<const std::string, uint16_t>> externals
  ) : internalOps(internals), externalOps(externals) {}
};

typedef std::unordered_map<dfcxx::OutputFormatID, std::string> DFOutputPaths;

struct DFOptionsConfig {
public:
  Scheduler scheduler;
  uint64_t stages;

  DFOptionsConfig() {
    scheduler = Scheduler::ASAP;
    stages = 0;
  }

  DFOptionsConfig(const DFOptionsConfig &) = default;

  std::string validate() const {
    if (scheduler != Scheduler::CombPipelining && stages > 0) {
      return "Pipeline stages cannot be specified without pipelining scheduler.";
    }

    if (scheduler == Scheduler::CombPipelining && stages == 0) {
      return "Pipeline stages were not specified along with pipelining sheduler.";
    }

    return "";
  }
};

} // namespace dfcxx

#endif // DFCXX_TYPEDEFS_H
