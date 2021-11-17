//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include <string>
#include <vector>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

enum BalanceMode {
  FlowSimple,
  FlowBlocking,
  LatencyLP,
  LatencyASAP,
  LatencyALAP
};

struct Buffer;

class LatencyBalancer {
public:
  LatencyBalancer() : graphTime(0) {}
  virtual ~LatencyBalancer() {}
  virtual void balance(Model &model) {}
  unsigned getGraphLatency() { return graphTime; }

protected:
  virtual void insertBuffers(Model &model) {};
  virtual void collectGraphTime() {};
  unsigned graphTime;
};



} // namespace eda::hls::scheduler
