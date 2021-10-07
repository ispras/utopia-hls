//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/library/library.h>

namespace eda::hls::library {

void VerilogPrinter::print(std::ostream &out) const
{
  out << "module " << type.name << "(" << std::endl;

  for (const eda::hls::model::Argument *arg: type.inputs) {
    out << "  input " << arg->name; // TODO print arg->length
    if (!(arg == type.inputs.back() && type.outputs.empty())) {
      out << "," << std::endl;
    }
  }

  for (const eda::hls::model::Argument *arg: type.outputs) {
    out << "  output " << arg->name; // TODO print arg->length
    if (arg != type.outputs.back()) {
      out << "," << std::endl;
    }
  }
  out << ");" << std::endl;
  out << "endmodule" << " // " << type.name << std::endl;
}

std::ostream& operator <<(std::ostream &out, const VerilogPrinter &printer) {
  printer.print(out);
  return out;
}

} // namespace eda::hls::library
