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

bool eqCheckTest(const std::string &fileM, const std::string &fileM2) {

  std::unique_ptr<Model> modelM = parse(fileM);
  std::unique_ptr<Model> modelM2 = parse(fileM2);
  Verifier verifier = Verifier::get();

  return verifier.equivalent(*modelM.get(), *modelM2.get());
}

TEST(DebuggerTest, Solve) {
  // TODO: wrong test, substitute 'true' by 'false'
  EXPECT_EQ(eqCheckTest("test/hil/test.hil", "test/hil/test_clone.hil"), true);
}

