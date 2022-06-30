//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/constraint.h"

#include <map>
#include <string>

namespace eda::hls::model {

//===----------------------------------------------------------------------===//
// Circuit Indicators
//===----------------------------------------------------------------------===//

enum Indicator {
  FREQ,
  PERF,
  TICKS,
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
  /// Outputs indicators.
  std::map<std::string, ChanInd> outputs;
};

using GraphInd   = NodeInd;
using ModelInd   = NodeInd;
using Indicators = NodeInd;

//===----------------------------------------------------------------------===//
// Optimization Criteria
//===----------------------------------------------------------------------===//

struct Criteria final {
  Criteria(Indicator objective,
           const Constraint<unsigned> &freq,
           const Constraint<unsigned> &perf,
           const Constraint<unsigned> &ticks,
           const Constraint<unsigned> &power,
           const Constraint<unsigned> &area):
    objective(objective),
    freq(freq),
    perf(perf),
    ticks(ticks),
    power(power),
    area(area) {}

  const Indicator objective;

  const Constraint<unsigned> freq;
  const Constraint<unsigned> perf;
  const Constraint<unsigned> ticks;
  const Constraint<unsigned> power;
  const Constraint<unsigned> area;

  /// Checks the constraints.
  bool check(const Indicators &indicators) const {
    return freq.check(indicators.freq())
        && perf.check(indicators.perf())
        && ticks.check(indicators.ticks)
        && power.check(indicators.power)
        && area.check(indicators.area);
  }
};

} // namespace eda::hls::model
