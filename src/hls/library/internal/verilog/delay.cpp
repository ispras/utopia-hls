//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/verilog/delay.h"

#include "util/string.h"

#include <cmath>

namespace eda::hls::library::internal::verilog {

void Delay::estimate(const Parameters &params, Indicators &indicators) const {
  unsigned widthSum = 0;

  unsigned width_value = params.getValue(width);

  widthSum += width_value * ports.size();

  double S = params.getValue(depth);
  double Areg = 16.0;
  double A = S * widthSum * Areg;
  double Fmax = 500000.0;
  double P = A;
  double D = 1000000000.0 / Fmax;

  indicators.ticks = static_cast<unsigned>(S);
  indicators.power = static_cast<unsigned>(P);
  indicators.area  = static_cast<unsigned>(A);
  indicators.delay = static_cast<unsigned>(D);

  ChanInd chanInd;
  chanInd.ticks = indicators.ticks;
  chanInd.delay = indicators.delay;

  indicators.outputs.clear();
  for (const auto &port : ports) {
    if (port.direction != Port::IN) {
      indicators.outputs.insert({ port.name, chanInd });
    }
  }
}

std::shared_ptr<MetaElement> Delay::create(const NodeType &nodetype,
                                           const HWConfig &hwconfig) {
    std::string name = nodetype.name;
    std::shared_ptr<MetaElement> metaElement;
    auto ports = createPorts(nodetype);
    std::string lowerCaseName = name;
    unsigned i = 0;
    while (lowerCaseName[i]) {
      lowerCaseName[i] = tolower(lowerCaseName[i]);
      i++;
    }
    Parameters params;
    params.add(Parameter(depth,
        Constraint<unsigned>(1, std::numeric_limits<unsigned>::max()), 1));
    params.add(Parameter(width,
        Constraint<unsigned>(1, std::numeric_limits<unsigned>::max()), 1));

    metaElement = std::shared_ptr<MetaElement>(new Delay(lowerCaseName,
                                                         "std",
                                                         false,
                                                         params,
                                                         ports));
  return metaElement;
};

std::unique_ptr<Element> Delay::construct() const {
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  std::string inputs, outputs, ifaceWires, regs, fsm, assigns;
  std::string outputType;

  outputType = std::string("reg ");

  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      ifaceWires += std::string("input ") + port.name + ";\n";
      continue;
    }

    std::string portDeclr =
      (port.width > 1 ? std::string("[") + std::to_string(port.width - 1)
                                         + ":0] " : std::string(""))
                                         + utils::replaceSomeChars(port.name) + ";\n";

    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      if (port.direction == Port::IN) {
        ifaceWires += std::string("input ") + portDeclr;
      } else {
        ifaceWires += std::string("inout ") + portDeclr;
      }
      inputs += std::string("wire ") + portDeclr;
    }

    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      if (port.direction == Port::OUT) {
        ifaceWires += std::string("output ") + portDeclr;
      }
      outputs += outputType + portDeclr;
    }
  }

  std::string ir;
  std::string inPort, outPort;
  unsigned d = 3; // FIXME
  regs += std::string("reg [31:0] state;\n");

  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      continue;
    }
    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      inPort = utils::replaceSomeChars(port.name);
    }
    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      outPort = utils::replaceSomeChars(port.name);
    }
  }

  ir += std::string(" if (state == 0) begin\n  state <= 1;\n  s0 <= ")
                    + inPort + "; end\nelse";
  regs += std::string("reg [31:0] s0;\n");
  for (unsigned i = 1; i < d; i++) {
    regs += std::string("reg [31:0] s") + std::to_string(i) + ";\n";
    ir += std::string(" if (state == ") + std::to_string(i) + ") begin\n";
    ir += std::string("  state <= ") + std::to_string(i + 1) + ";\n";
    ir += std::string("  s") + std::to_string(i) + " <= s"
                             + std::to_string(i - 1) + "; end\nelse";
  }
  ir += std::string(" begin\n  state <= 0;\n  ") + outPort + " <= s"
                    + std::to_string(d - 1) + "; end\nend\n";
  regs += std::string("always @(posedge clock) begin\nif (!reset) begin\n");
  regs += std::string("state <= 0; end\nelse");
  element->ir = std::string("\n") + ifaceWires + inputs + outputs + regs + ir;
  return element;
}

bool Delay::isDelay(const NodeType &nodeType) {
   return nodeType.isDelay();
}

} // namespace eda::hls::library::internal::verilog