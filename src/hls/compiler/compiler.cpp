//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <memory>

#include <hls/compiler/compiler.h>
#include <hls/library/library.h>

using namespace eda::hls::library;

namespace eda::hls::compiler {

void Compiler::print(std::ostream &out) const {
  // print model.nodetypes
  for (const auto *nodetype : model.nodetypes) {
    auto printer = std::make_unique<VerilogNodeTypePrinter>(*nodetype);
    out << *printer;
  }

  // print model.graphs
  for (const auto *graph : model.graphs) {
    auto printer = std::make_unique<VerilogGraphPrinter>(*graph);
    out << *printer;
  }
}

std::ostream& operator <<(std::ostream &out, const Compiler &compiler) {
  compiler.print(out);
  return out;
}

} // namespace eda::hls::compiler
