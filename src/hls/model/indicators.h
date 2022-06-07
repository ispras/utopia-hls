//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

namespace eda::hls::model {

enum Indicator {
  FREQ,
  PERF,
  LATENCY,
  POWER,
  AREA
};

/// Channel indicators.
struct ChanInd {
  /// Latency (ticks).
  unsigned latency;
  /// Combinational delay (ns).
  unsigned delay;
};

/// Node indicators.
struct NodeInd {
  /// Frequency (Hz).
  unsigned freq;
  /// Throughput (ops).
  unsigned perf;
  /// Power (mW).
  unsigned power;
  /// Area (cells).
  unsigned area;
  /// Latency (ticks): maximum over all output channels (see below).
  unsigned latency;
  /// Combinational delay (ns): maximum over all output channels (see below).
  unsigned delay;
  /// Outputs indicators.
  std::map<std::string, ChanInd> outputs;
};

using GraphInd = NodeInd;
using ModelInd = NodeInd;

using Indicators = NodeInd;

} // namespace eda::hls::model
