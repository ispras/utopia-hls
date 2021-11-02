//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"
#include "util/assert.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace eda::hls::library {

void VerilogNodeTypePrinter::print(std::ostream &out) const {
  out << "module " << type.name << "(" << std::endl;

  bool comma = false;
  for (const auto *arg: type.inputs) {
    out << (comma ? ", " : "");
    out << arg->name;
    comma = true;
  }

  for (const auto *arg: type.outputs) {
    out << (comma ? ", " : "");
    out << arg->name;
    comma = true;
  }
  out << ");" << std::endl;

  auto meta = Library::get().find(type);
  // meta.params.value["f"] can be adjusted before calling construct
  // meta.params.set(std::string("f"), 3);
  auto element = Library::get().construct(meta);
  out << element->ir << std::endl;

  out << "endmodule" << " // " << type.name << std::endl;
}

void VerilogGraphPrinter::printChan(std::ostream &out, const eda::hls::model::Chan &chan) const {
  // TODO: chan.type is ignored
  out << "." << chan.source.node->name << "(" << chan.target.node->name << ")";
}

void VerilogGraphPrinter::print(std::ostream &out) const {
  out << "module " << graph.name << "();" << std::endl;

  for (const auto *chan : graph.chans) {
    out << "wire " << chan->name << ";" << std::endl;
  }
  for (const auto *node : graph.nodes) {
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
  out << "endmodule // " << graph.name << std::endl;
}

std::ostream& operator <<(std::ostream &out, const VerilogNodeTypePrinter &printer) {
  printer.print(out);
  return out;
}

std::ostream& operator <<(std::ostream &out, const VerilogGraphPrinter &printer) {
  printer.print(out);
  return out;
}

const MetaElement& Library::find(const std::string &name) const {
  const auto i = std::find_if(library.begin(), library.end(),
    [&name](const MetaElement &meta) { return meta.name == name; });
  uassert(i != library.end(), "Current version of library doesn't include element " << name);
  return *i;
}

const MetaElement& Library::find(const eda::hls::model::NodeType &type) {
  const auto i = std::find_if(library.begin(), library.end(),
    [&type](const MetaElement &meta) { return meta.name == type.name; });
  /// TODO Dynamic population of the library should be prohibited
  if (i == library.end()) {
    library.push_back(createMetaElement(type));
    return library.back();
  }
  return *i;
}

MetaElement Library::createMetaElement(const eda::hls::model::NodeType &type) const {
  Parameters params("");
  Ports ports;

  params.add(Parameter("f", Constraint(1, 1000), 100));

  /// Copy ports from model
  for (const auto *arg: type.inputs) {
    ports.push_back(Port(arg->name, Port::IN, arg->latency, 1 /*arg->length*/));
  }
  for (const auto *arg: type.outputs) {
    ports.push_back(Port(arg->name, Port::OUT, arg->latency, 1 /*arg->length*/));
  }

  return MetaElement(type.name, params, ports);
}

MetaElement Library::createMetaElement(const std::string &name) const {
  Parameters params("");
  Ports ports;

  params.add(Parameter("f", Constraint(1, 1000), 100));

  /// Populate ports for different library elements
  ports.push_back(Port("clock", Port::IN, 0, 1));
  ports.push_back(Port("reset", Port::IN, 0, 1));

  if (name == "merge") {
    ports.push_back(Port("in1", Port::IN, 0, 1));
    ports.push_back(Port("in2", Port::IN, 0, 1));
    ports.push_back(Port("out", Port::OUT, 1, 1));
  } else if (name == "split") {
    ports.push_back(Port("in", Port::IN, 0, 1));
    ports.push_back(Port("out1", Port::OUT, 1, 1));
    ports.push_back(Port("out2", Port::OUT, 1, 1));
  } else if (name == "delay") {
    ports.push_back(Port("in", Port::IN, 0, 1));
    ports.push_back(Port("out", Port::OUT, 1, 1));
  } else if (name == "add" || name == "sub") {
    ports.push_back(Port("a", Port::IN, 0, 4));
    ports.push_back(Port("b", Port::IN, 0, 4));
    ports.push_back(Port("c", Port::OUT, 2, 4));
    ports.push_back(Port("d", Port::OUT, 2, 1));
  } else {
    uassert(false, "Call createMetaElement by NodeType for element " << name);
  }

  return MetaElement(name, params, ports);
}

std::unique_ptr<Element> Library::construct(const MetaElement &meta) const {
  std::unique_ptr<Element> element = std::make_unique<Element>(meta.ports);
  std::string inputs, outputs, iface_wires;
  unsigned int pos = 0, input_length = 0, output_length = 0;

  for (auto port : meta.ports) {
    if (port.name != "clock" && port.name != "reset") {
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        input_length += port.width;
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        output_length += port.width;
      }
    }
  }

  if (output_length == 0) {
    element->ir = iface_wires;
    return element;
  }

  bool first_port = true;
  for (auto port : meta.ports) {
    if (port.name == "clock" || port.name == "reset") {
      iface_wires += std::string("input ") + port.name + ";\n";
      continue;
    }

    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      if (port.direction == Port::IN) {
        iface_wires += std::string("input [") + std::to_string(port.width - 1) + ":0] " + port.name + ";\n";
      } else {
        iface_wires += std::string("inout [") + std::to_string(port.width - 1) + ":0] " + port.name + ";\n";
      }
      if (first_port) {
        inputs += std::string("reg [") + std::to_string(input_length - 1) + ":0] stage_0 = {";
        first_port = false;
      } else {
        inputs += std::string(", ");
      }
      inputs += port.name;
    }

    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      iface_wires += std::string("output [") + std::to_string(port.width - 1) + ":0] " + port.name + ";\n";
      outputs += std::string("wire [") + std::to_string(port.width - 1) + ":0] " + port.name +
                 " = stage_" + std::to_string(port.latency - 1) +
                 "[" + std::to_string((pos + port.width - 1) % output_length) + ":" +
                       std::to_string(pos % output_length) + "];\n";
      pos += port.width;
    }
  }

  inputs += std::string("};\n");

  /// Extract frequency.
  unsigned f = meta.params.value("f");
  for (unsigned i = 1; i < f; i++) {
    if (input_length > 2) {
      inputs += std::string("reg [") + std::to_string(input_length - 1) + ":0] state_" + std::to_string(i) +
                            " = {state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 2) + ":0], " +
                            "state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 1) + "]};\n";
    } else {
      inputs += std::string("state_") + std::to_string(i) + "[1:0] = state_" + std::to_string(i - 1) + "[0:1];\n";
    }
  }

  element->ir = iface_wires + inputs + outputs;
  return element;
}

void Library::estimate(const Parameters &params, Indicators &indicators) const {
  // TODO: estimations of l, f, p, a
  indicators.frequency = params.value("f");
  indicators.throughput = indicators.frequency;
  indicators.latency = 1000000;
  indicators.power = 5;
  indicators.area = 10000;
}

Library::Library() {
  library.push_back(createMetaElement("merge"));
  library.push_back(createMetaElement("split"));
  library.push_back(createMetaElement("delay"));
  library.push_back(createMetaElement("add"));
  library.push_back(createMetaElement("sub"));
}

} // namespace eda::hls::library
