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

#include <limits>

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::scheduler;

void paramOptimizerTest(const std::string &filename) {
  // Optimization criterion and constraints.
  Criteria criteria(Throughput,
    Constraint(1000, 500000),                               // Frequency (kHz)
    Constraint(1000, 500000),                               // Throughput (=frequency)
    Constraint(0,    100),                                  // Latency (cycles)
    Constraint(0,    std::numeric_limits<unsigned>::max()), // Power (does not matter)
    Constraint(1,    150000));                              // Area (number of LUTs)

  // Model whose parameters need to be optimized.
  std::unique_ptr<Model> model = parse(filename);

  // Integral indicators of the optimized model (output).
  Indicators indicators;

  // Optimize parameters.
  std::map<std::string, Parameters> params =
    ParametersOptimizer::get().optimize(criteria, *model, indicators);

  // Check the constrains.
  EXPECT_TRUE(criteria.check(indicators));
}

TEST(SchedulerTest, ParamOptimizerBase) {
  paramOptimizerTest("test/data/hil/test.hil");
}

TEST(SchedulerTest, ParamOptimizerIDCT) {
  paramOptimizerTest("test/data/hil/idct.hil");
}
