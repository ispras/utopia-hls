//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <hls/scheduler/scheduler.h>

using namespace eda::hls::model;

namespace eda::hls::scheduler {

class DijkstraBalancer final : public LatencyBalancer {
public:
  DijkstraBalancer(Model* model_arg) : LatencyBalancer(model_arg) { }


};

} // namespace eda::hls::scheduler
