//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/param_optimizer.h"

#include "gtest/gtest.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

void paramOptimizerTest(const std::string &filename) {
  // Optimization criterion and constraints.
  Criteria criteria(Throughput,
    Constraint(1, 1000),  // Frequency
    Constraint(1, 1000),  // Throughput
    Constraint(0, 1000),  // Latency
    Constraint(0, 1000),  // Power
    Constraint(1, 1000)); // Area

  // Model whose parameters need to be optimized.
  std::unique_ptr<Model> model = parse(filename);

  // Integral indicators of the optimized model (output).
  Indicators indicators;

  // Optimize parameters.
  std::map<std::string, Parameters> params =
    ParametersOptimizer::get().optimize(*model, criteria, indicators);

  // Check the constrains.
  EXPECT_TRUE(criteria.frequency.min <= indicators.frequency &&
              criteria.frequency.max >= indicators.frequency);
  EXPECT_TRUE(criteria.throughput.min <= indicators.throughput &&
              criteria.throughput.max >= indicators.throughput);
  EXPECT_TRUE(criteria.latency.min <= indicators.latency &&
              criteria.latency.max >= indicators.latency);
  EXPECT_TRUE(criteria.power.min <= indicators.power &&
              criteria.power.max >= indicators.power);
  EXPECT_TRUE(criteria.area.min <= indicators.area &&
              criteria.area.max >= indicators.area);
}

TEST(SchedulerTest, ParamOptimizerSimple) {
  paramOptimizerTest("test/data/hil/test.hil");
}

TEST(SchedulerTest, ParamOptimizerIDCT) {
  paramOptimizerTest("test/data/hil/idct.hil");
}
