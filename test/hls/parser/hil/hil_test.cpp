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
#include "hls/library/library.h"
#include "hls/model/model.h"
#include "hls/model/printer.h"
#include "hls/parser/hil/parser.h"

using namespace eda::hls::compiler;
using namespace eda::hls::library;
using namespace eda::hls::model;
using namespace eda::hls::parser::hil;

int hilTest(const std::string &filename) {
  std::unique_ptr<Model> model = parse(filename);

  if (!model)
    return -1;

  std::cout << *model;
  printDot(std::cout, *model);

  return 0;
}

int hilTestNodeTypes(const std::string &filename) {
  return (parse(filename))->nodetypes.size();
}

int hilTestGraphs(const std::string &filename) {
  return (parse(filename))->graphs.size();
}

int hilTestVerilogNodeTypePrinter(const std::string &filename) {
  auto library = std::make_unique<Library>();
  auto nodetypes = parse(filename)->nodetypes;

  std::cout << "------ Verilog RTL-model ------" << std::endl;
  for (const auto *nodetype: nodetypes) {
    auto printer = std::make_unique<VerilogNodeTypePrinter>(*nodetype, *library);
    std::cout << *printer;
  }

  return 0;
}

int hilTestCompiler(const std::string &filename) {
  auto library = std::make_unique<Library>();
  auto compiler = std::make_unique<Compiler>(*parse(filename), *library);
  std::cout << *compiler;

  ElementArguments ea(std::string("add"));
  MetaElementDescriptor med = library->find(ea.name);
  Parameter param = med.parameters[0];
  unsigned f = (param.constraint.hiValue - param.constraint.loValue) >> 1;
  ea.args.insert(std::pair<std::string, unsigned>("f", f));

  auto element = library->construct(ea);
  std::cout << element->ir << std::endl;
  return 0;
}

TEST(HilTest, SimpleTest) {
  EXPECT_EQ(hilTest("test/data/hil/test.hil"), 0);
  EXPECT_EQ(hilTest("test/data/hil/idct.hil"), 0);
}

TEST(HilTest, NodeTypesTest) {
  EXPECT_EQ(hilTestNodeTypes("test/data/hil/test.hil"), 6);
}

TEST(HilTest, GraphsTest) {
  EXPECT_EQ(hilTestGraphs("test/data/hil/test.hil"), 1);
}

TEST(HilTest, VerilogNodeTypePrinterTest) {
  EXPECT_EQ(hilTestVerilogNodeTypePrinter("test/data/hil/test.hil"), 0);
}

TEST(HilTest, CompilerTest) {
  EXPECT_EQ(hilTestCompiler("test/data/hil/test.hil"), 0);
}

