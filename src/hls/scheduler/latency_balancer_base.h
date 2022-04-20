//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains the declaration of the LatencyBalancerBase class, that 
/// is the base class for all schedulers.
///
/// \author <a href="mailto:lebedev@ispras.ru">Mikhail Lebedev</a>
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include <string>
#include <vector>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class LatencyBalancerBase {
public:
  LatencyBalancerBase() : graphTime(0) {}
  virtual ~LatencyBalancerBase() {}

  /// Schedules the specified model.
  virtual void balance(Model &model) = 0; 

  /// Returns the maximum latency of the main graph.
  unsigned getGraphLatency() { return graphTime; }

protected:
  /// Inserts the balancing buffers into the model.
  virtual void insertBuffers(Model &model) = 0;

  /// Computes the maximum latency of the main graph.
  virtual void collectGraphTime() = 0;

  unsigned graphTime;
};



} // namespace eda::hls::scheduler
