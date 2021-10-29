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

#include "hls/model/model.h"
#include "hls/compiler/compiler.h"
#include "hls/library/library.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::compiler;
using namespace eda::hls::library;
using namespace eda::hls::model;
using namespace eda::hls::parser::hil;

int hilTest(const std::string &filename) {
  std::cout << *parse(filename);
  return 0;
}

int hilTestNodeTypes(const std::string &filename) {
  return (parse(filename))->nodetypes.size();
}

int hilTestGraphs(const std::string &filename) {
  return (parse(filename))->graphs.size();
}

int hilTestVerilogNodeTypePrinter(const std::string &filename) {
  auto nodetypes = parse(filename)->nodetypes;

  std::cout << "------ Verilog RTL-model ------" << std::endl;
  for (const auto *nodetype: nodetypes) {
    auto printer = std::make_unique<VerilogNodeTypePrinter>(*nodetype);
    std::cout << *printer;
  }

  return 0;
}

int hilTestCompiler(const std::string &filename) {
  auto compiler = std::make_unique<Compiler>(*parse(filename));
  std::cout << *compiler;

  auto library = std::make_unique<Library>();
  MetaElementDescriptor med = library->find(std::string("add"));
  ElementArguments ea;
  for (auto e : med.parameters) {
    ea.push_back(e.port);
  }
  auto element = library->construct(ea, 3);
  std::cout << element->ir << std::endl;
  return 0;
}

TEST(HilTest, SimpleTest) {
  EXPECT_EQ(hilTest("test/hil/test.hil"), 0);
  EXPECT_EQ(hilTest("test/hil/idct.hil"), 0);
}

TEST(HilTest, NodeTypesTest) {
  EXPECT_EQ(hilTestNodeTypes("test/hil/test.hil"), 6);
}

TEST(HilTest, GraphsTest) {
  EXPECT_EQ(hilTestGraphs("test/hil/test.hil"), 1);
}

TEST(HilTest, VerilogNodeTypePrinterTest) {
  EXPECT_EQ(hilTestVerilogNodeTypePrinter("test/hil/test.hil"), 0);
}

TEST(HilTest, CompilerTest) {
  EXPECT_EQ(hilTestCompiler("test/hil/test.hil"), 0);
}

