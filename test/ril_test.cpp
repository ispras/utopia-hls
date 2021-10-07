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

#include "gate/model/netlist.h"
#include "rtl/compiler/compiler.h"
#include "rtl/library/flibrary.h"
#include "rtl/model/net.h"
#include "rtl/parser/ril/builder.h"
#include "rtl/parser/ril/parser.h"

using namespace eda::gate::model;
using namespace eda::rtl::compiler;
using namespace eda::rtl::library;
using namespace eda::rtl::model;
using namespace eda::rtl::parser::ril;

int ril_test(const std::string &filename) {
  auto model = parse(filename);

  std::cout << "------ p/v-nets ------" << std::endl;
  std::cout << *model << std::endl;

  Compiler compiler(FLibraryDefault::get());
  auto netlist = compiler.compile(*model);

  std::cout << "------ netlist ------" << std::endl;
  std::cout << *netlist;

  return 0;
}

TEST(RilTest, SingleTest) {
  EXPECT_EQ(ril_test("test/ril/test.ril"), 0);
}
