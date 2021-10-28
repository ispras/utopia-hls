//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <memory>
#include <stdexcept>

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

const MetaElementDescriptor& Library::find(const std::string &name) const {
  unsigned i = 0;
  for (; i < library.size(); i++) {
    if (library[i].name == name) {
      return library[i];
    }
  }

  throw std::runtime_error(std::string("Current version of library doesn't include element ") + name);
}

std::shared_ptr<ElementDescriptor> Library::construct(const ElementArguments &args, unsigned f) const {
  std::shared_ptr<ElementDescriptor> ed = std::make_shared<ElementDescriptor>(args);
  std::string ir;

  for (auto port : args) {
    ir += std::string(port.direction == Port::IN ? "input" : port.direction == Port::OUT ? "output" : "inout") + " " + 
           port.name + ": " + std::to_string(port.width) + " f=" + std::to_string(f);
  }
  // TODO: implementation of the scheme by using f.
  ed->ir = ir;

  return ed;
}

std::shared_ptr<ElementCharacteristics> Library::estimate(const ElementArguments &args) const {
  ExtendedElementArguments latencies;

  // TODO: estimations of l, f, p, a
  for (const auto &p : args) {
    latencies.push_back(ExtendedPort(p, 1));
  }

  unsigned frequency = 100;
  unsigned power = 5;
  unsigned area = 10000;

  std::shared_ptr<ElementCharacteristics> ec =
    std::make_shared<ElementCharacteristics>(latencies, frequency, power, area);

  return ec;
}

Library::Library() {
  Parameters p;

  p.push_back(Parameter(Port("clock", Port::IN, 1), Constraint(1, 1)));
  p.push_back(Parameter(Port("reset", Port::IN, 1), Constraint(1, 1)));
  p.push_back(Parameter(Port("in", Port::IN, 3), Constraint(2, 4)));
  p.push_back(Parameter(Port("out", Port::OUT, 2), Constraint(1, 3)));

  library.push_back(MetaElementDescriptor("add", p));
  library.push_back(MetaElementDescriptor("sub", p));
}

} // namespace eda::hls::library
