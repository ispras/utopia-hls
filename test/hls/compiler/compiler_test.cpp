//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>

#include "gtest/gtest.h"

#include "hls/compiler/compiler.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::compiler;
using namespace eda::hls::parser::hil;

int compileSimpleHilTest(const std::string &filename) {
  auto compiler = std::make_unique<Compiler>(*parse(filename));
  std::cout << *compiler;

  return 0;
}

TEST(CompilerTest, CompileTestHilTest) {
  EXPECT_EQ(compileSimpleHilTest("test/data/hil/test.hil"), 0);
}

