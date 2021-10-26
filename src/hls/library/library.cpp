//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/library/library.h>

namespace eda::hls::library {

void VerilogNodeTypePrinter::print(std::ostream &out) const
{
  out << "module " << t.name << "(" << std::endl;

  for (const auto *arg: t.inputs) {
    out << "  input " << arg->name; // TODO print arg->length
    if (!(arg == t.inputs.back() && t.outputs.empty())) {
      out << "," << std::endl;
    }
  }

  for (const auto *arg: t.outputs) {
    out << "  output " << arg->name; // TODO print arg->length
    if (arg != t.outputs.back()) {
      out << "," << std::endl;
    }
  }
  out << ");" << std::endl;
  out << "endmodule" << " // " << t.name << std::endl;
}

void VerilogGraphPrinter::printChan(std::ostream &out, const eda::hls::model::Chan &chan) const {
  // TODO: chan.type is ignored
  out << "." << chan.source.node->name << "(" << chan.target.node->name << ")";
}

void VerilogGraphPrinter::print(std::ostream &out) const
{
  out << "module " << g.name << "();" << std::endl;

  for (const auto *chan : g.chans) {
    out << "wire " << chan->name << ";" << std::endl;
  }
  for (const auto *node : g.nodes) {
    // TODO: node.type is ignored
    out << node->type.name << " " << node->name << "(";

    bool comma = false;

    for (const auto *input : node->inputs) {
      out << (comma ? ", " : "");
      printChan(out, *input);
      comma = true;
    }

    for (const auto *output: node->outputs) {
      out << (comma ? ", " : "");
      printChan(out, *output);
      comma = true;
    }
    out << ");" << std::endl;
  }
  out << "endmodule // " << g.name << std::endl;
}

std::ostream& operator <<(std::ostream &out, const VerilogNodeTypePrinter &printer) {
  printer.print(out);
  return out;
}

std::ostream& operator <<(std::ostream &out, const VerilogGraphPrinter &printer) {
  printer.print(out);
  return out;
}

} // namespace eda::hls::library
