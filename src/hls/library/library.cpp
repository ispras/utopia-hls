//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/library/library.h>

namespace eda::hls::library {

std::ostream& operator <<(std::ostream &out, const VerilogPrinter &printer)
{
  out << "module " << printer.type.name << "(" << std::endl;

  for (const eda::hls::model::Argument *arg: printer.type.inputs) {
    out << "  input " << arg->name; // TODO print arg->length
    if (!(arg == printer.type.inputs.back() && printer.type.outputs.empty())) {
      out << "," << std::endl;
    }
  }

  for (const eda::hls::model::Argument *arg: printer.type.outputs) {
    out << "  output " << arg->name; // TODO print arg->length
    if (arg != printer.type.outputs.back()) {
      out << "," << std::endl;
    }
  }
  out << ");" << std::endl;
  out << "endmodule" << " // " << printer.type.name << std::endl;

  return out;
}

} // namespace eda::hls::library
