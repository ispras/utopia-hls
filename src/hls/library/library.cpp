//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/library.h"

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace eda::hls::library {

void VerilogNodeTypePrinter::print(std::ostream &out) const {
  //Ports ports;
  //ports.push_back(Port("clock", Port::IN, 1));
  //ports.push_back(Port("reset", Port::IN, 1));

  out << "module " << type.name << "(" << std::endl;

  bool comma = false;
  for (const auto *arg: type.inputs) {
    out << (comma ? ", " : "");
    out << arg->name;
    comma = true;
    //ports.push_back(Port(arg->name, Port::IN, 1 /*arg->length*/));
  }

  for (const auto *arg: type.outputs) {
    out << (comma ? ", " : "");
    out << arg->name;
    comma = true;
    //ports.push_back(Port(arg->name, Port::OUT, 1 /*arg->length*/));
  }
  out << ");" << std::endl;

  ElementArguments ea(type.name);
  ea.args.insert(std::pair<std::string, unsigned>("f", 3));
  auto element = library.construct(ea);
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

const MetaElementDescriptor& Library::find(const std::string &name) const {
  unsigned i = 0;
  for (; i < library.size(); i++) {
    if (library[i].name == name) {
      return library[i];
    }
  }

  throw std::runtime_error(std::string("Current version of library doesn't include element ") + name);
}

std::unique_ptr<ElementDescriptor> Library::construct(const ElementArguments &args) const {
  Ports ports;

  /// Populate ports for different library elements
  ports.push_back(Port("clock", Port::IN, 0, 1));
  ports.push_back(Port("reset", Port::IN, 0, 1));
  if (args.name == "merge") {
    ports.push_back(Port("in1", Port::IN, 0, 1));
    ports.push_back(Port("in2", Port::IN, 0, 1));
    ports.push_back(Port("out", Port::OUT, 1, 1));
  } else if (args.name == "split") {
    ports.push_back(Port("in", Port::IN, 0, 1));
    ports.push_back(Port("out1", Port::OUT, 1, 1));
    ports.push_back(Port("out2", Port::OUT, 1, 1));
  } else if (args.name == "delay") {
    ports.push_back(Port("in", Port::IN, 0, 1));
    ports.push_back(Port("out", Port::OUT, 1, 1));
  } else if (args.name == "add" || args.name == "sub") {
    ports.push_back(Port("a", Port::IN, 0, 4));
    ports.push_back(Port("b", Port::IN, 0, 4));
    ports.push_back(Port("c", Port::OUT, 2, 4));
    ports.push_back(Port("d", Port::OUT, 2, 1));
  }

  std::unique_ptr<ElementDescriptor> ed = std::make_unique<ElementDescriptor>(ports);
  std::string inputs, outputs, iface_wires;
  unsigned int pos = 0, input_length = 0, output_length = 0;

  for (auto port : ports) {
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
    ed->ir = iface_wires;
    return ed;
  }

  bool first_port = true;
  for (auto port : ports) {
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
  unsigned f = args.args.find("f")->second;
  for (unsigned int i = 1; i < f; i++) {
    if (input_length > 2) {
      inputs += std::string("reg [") + std::to_string(input_length - 1) + ":0] state_" + std::to_string(i) +
                            " = {state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 2) + ":0], " +
                            "state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 1) + "]};\n";
    } else {
      inputs += std::string("state_") + std::to_string(i) + "[1:0] = state_" + std::to_string(i - 1) + "[0:1];\n";
    }
  }

  ed->ir = iface_wires + inputs + outputs;
  return ed;
}

std::unique_ptr<ElementCharacteristics> Library::estimate(const ElementArguments &args) const {
  // TODO: estimations of l, f, p, a
  unsigned frequency = args.args.find("f")->second;
  unsigned throughput = frequency;
  unsigned power = 5;
  unsigned area = 10000;

  std::unique_ptr<ElementCharacteristics> ec =
    std::make_unique<ElementCharacteristics>(frequency, throughput, power, area);

  return ec;
}

Library::Library() {
  Parameters params;

  params.push_back(Parameter("f", Constraint(1, 10)));
  library.push_back(MetaElementDescriptor("merge", params));
  library.push_back(MetaElementDescriptor("split", params));
  library.push_back(MetaElementDescriptor("delay", params));
  library.push_back(MetaElementDescriptor("add", params));
  library.push_back(MetaElementDescriptor("sub", params));
}

} // namespace eda::hls::library
