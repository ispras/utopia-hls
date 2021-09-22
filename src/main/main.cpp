/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#include <iostream>

#include "gate/model/gate.h"
#include "gate/model/netlist.h"
#include "rtl/compiler/compiler.h"
#include "rtl/library/flibrary.h"
#include "rtl/model/net.h"
#include "rtl/parser/builder.h"
#include "rtl/parser/parser.h"

using namespace eda::gate::model;
using namespace eda::rtl::compiler;
using namespace eda::rtl::library;
using namespace eda::rtl::model;
using namespace eda::rtl::parser;

int main(int argc, char **argv) {
  std::cout << "EDA Utopia | Copyright (c) 2021 ISPRAS" << std::endl;

  if (argc <= 1) {
    std::cout << "Usage: " << argv[0] << " <input-file>" << std::endl;
    std::cout << "Synthesis terminated." << std::endl;
    return -1;
  }

  const std::string filename = argv[1];
  if (parse(filename) == -1) {
    std::cout << "Could not parse " << filename << std::endl;
    std::cout << "Synthesis terminated." << std::endl;
    return -1;
  }

  std::unique_ptr<Net> pnet = Builder::get().create();
  pnet->create();

  std::cout << "------ p/v-nets ------" << std::endl;
  std::cout << *pnet << std::endl;

  Compiler compiler(FLibraryDefault::get());
  std::unique_ptr<Netlist> netlist = compiler.compile(*pnet);

  std::cout << "------ netlist ------" << std::endl;
  std::cout << *netlist;

  return 0;
}

