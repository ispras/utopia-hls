//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/verilog/shr8.h"

#include "utils/string.h"

#include <cmath>

namespace eda::hls::library::internal::verilog {

void Shr8::estimate(const Parameters &params, Indicators &indicators) const {
  double S = params.getValue(stages);
  double Fmax = 500000.0;
  double F = Fmax * ((1 - std::exp(-S / 10.0)) + 0.1);
  double Sa = 100.0 * ((double)std::rand() / RAND_MAX) + 1;
  double W = params.getValue(width);
  double A = 100.0 * (1.0 - std::exp(-(S - Sa) * (S - Sa) / 4.0)) * W;
  double P = A;
  double D = 1000000000.0 / F;

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

std::shared_ptr<MetaElement> Shr8::create(const NodeType &nodetype,
                                         const HWConfig &hwconfig) {
  std::string name = nodetype.name;
  auto ports = createPorts(nodetype);
  std::string lowerCaseName = name;
  unsigned i = 0;
  while (lowerCaseName[i]) {
    lowerCaseName[i] = tolower(lowerCaseName[i]);
    i++;
  }
  Parameters params;
  params.add(Parameter(stages, Constraint<unsigned>(1, 100), 0));
  params.add(Parameter(width, Constraint<unsigned>(1, 128), 16));
  return std::make_shared<Shr8>(lowerCaseName, "std", true, params, ports);
}

std::unique_ptr<Element> Shr8::construct() const {
  auto element = std::make_unique<Element>(ports);
  std::string inputs, outputs, ifaceWires, regs, fsm, assigns;
  std::string outputType;

  outputType = std::string("wire ");

  for (const auto &port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      ifaceWires += std::string("input ") + port.name + ";\n";
      continue;
    }

    std::string portDeclr =
        (port.width > 1 ? std::string("[") + std::to_string(port.width - 1) +
        ":0] " : std::string("")) + utils::replaceSomeChars(port.name) + ";\n";

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
      } else {
        ifaceWires += std::string("inout ") + portDeclr;
      }
      outputs += outputType + portDeclr;
    }
  }
  std::string ir;
  std::string inPortName, outPortName;
  for (const auto &port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      continue;
    }
    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      inPortName = utils::replaceSomeChars(port.name);
    }
    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      outPortName = utils::replaceSomeChars(port.name);
    }
  }

  ir += "assign " + outPortName + " = " + "$signed(" + inPortName + 
      ")" + " >>> 8";
  ir += ";\n";
  element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
  return element;
}

bool Shr8::isShr8(const NodeType &nodeType) {
  return nodeType.outputs.size() == 1 
      && nodeType.inputs.size()  == 1
      && utils::starts_with(nodeType.name, "SHR8");
}

} // namespace eda::hls::library::internal::verilog