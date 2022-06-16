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
  unsigned ticks = 0;
  /// Combinational delay (ps).
  unsigned delay = 0;
};

/// Node indicators.
struct NodeInd {
  /// Frequency (kHz).
  unsigned freq() const { return 1000000000 / delay; }
  /// Average frequency (kHz).
  unsigned avgFreq() const { return 1000000000 / averageDelay; }
  /// Throughput (kops).
  unsigned perf() const { return freq(); }

  /// Power (mW).
  unsigned power = 0;
  /// Area (cells).
  unsigned area = 0;
  /// Latency (ticks): maximum over all output channels (see below).
  unsigned ticks = 0;
  /// Combinational delay (ps): maximum over all output channels (see below).
  unsigned delay = 0;
  /// Average delay (ps).
  unsigned averageDelay = 0; // FIXME:
  /// Outputs indicators.
  std::map<std::string, ChanInd> outputs;
};

using GraphInd = NodeInd;
using ModelInd = NodeInd;

using Indicators = NodeInd;

} // namespace eda::hls::model
