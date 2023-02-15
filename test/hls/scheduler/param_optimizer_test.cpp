//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/mapper/mapper.h"
#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"
#include "hls/scheduler/param_optimizer.h"
#include "hls/scheduler/topological_balancer.h"

#include "gtest/gtest.h"

#include <limits>

namespace tool = eda::hls;
namespace mdl = tool::model;
namespace sched = tool::scheduler;

void paramOptimizerTest(const std::string &filename) {
  // Optimization criterion and constraints.
  mdl::Criteria criteria(
    PERF,
    mdl::Constraint<unsigned>(1000, 500000),  // Frequency (kHz)
    mdl::Constraint<unsigned>(1000, 500000),  // Performance (=frequency)
    mdl::Constraint<unsigned>(0,    100),     // Latency (cycles)
    mdl::Constraint<unsigned>(),              // Power (does not matter)
    mdl::Constraint<unsigned>(1,    150000)); // Area (number of LUTs)

  // Model whose parameters need to be optimized.
  std::shared_ptr<mdl::Model> model = tool::parser::hil::parse(filename);

  // Map model nodes to meta elements.
  tool::mapper::Mapper::get().map(*model, tool::library::Library::get());

  // Integral indicators of the optimized model (output).
  mdl::Indicators indicators;

  // Optimize parameters.
  std::map<std::string, mdl::Parameters> params 
    = sched::ParametersOptimizer<sched::TopologicalBalancer>::get()
      .optimize(criteria, *model, indicators);

  // Check the constrains.
  EXPECT_TRUE(criteria.check(indicators));
}

/*TEST(SchedulerTest, ParamOptimizerBase) {
  paramOptimizerTest("test/data/hil/test.hil");
}*/

/*TEST(SchedulerTest, ParamOptimizerIDCT) {
  paramOptimizerTest("test/data/hil/idct.hil");
}*/
