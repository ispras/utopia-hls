//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "hls/model/model.h"
#include "hls/parser/hil/parser.h"
#include "hls/debugger/debugger.h"

using namespace eda::hls::model;
using namespace eda::hls::parser::hil;
using namespace eda::hls::debugger;

bool eqCheckerTest(const std::string &filename) {

  std::unique_ptr<Model> model = parse(filename);
  std::unique_ptr<Model> modelClone = parse(filename);
  Verifier verifier = Verifier::get();

  return verifier.equivalent(*model.get(), *modelClone.get());
}

TEST(DebuggerTest, Solve) {
  EXPECT_EQ(eqCheckerTest("test/hil/test.hil"), false);
}

