//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/verilog/cast.h"

#include "utils/string.h"

#include <cmath>

namespace eda::hls::library::internal::verilog {

void Cast::estimate(const Parameters &params, Indicators &indicators) const {

  double S = params.getValue(stages);
  double Fmax = 500000.0;
  double F = Fmax * ((1 - std::exp(-S / 60.0)) + 0.1);
  double Sa = 100.0 * ((double)std::rand() / RAND_MAX) + 1;
  double W = params.getValue(width);
  double A = 100.0 * (1.0 - std::exp(15 * -(S - Sa) * (S - Sa) / 4.0)) * W;
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

std::shared_ptr<MetaElement> Cast::create(const NodeType &nodetype,
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
  params.add(Parameter(stages, Constraint<unsigned>(1, 100), 20));
  params.add(Parameter(width, Constraint<unsigned>(1, 128), 16));
  return std::make_shared<Cast>(lowerCaseName, "std", false, params, ports);
}

std::unique_ptr<Element> Cast::construct() const {
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
        ":0] " : std::string("")) + utils::replaceSomeChars(port.name) +
        ";\n";

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
  std::string inPortName;
  int inPortWidth = 0;
  std::string outPortName;
  int outPortWidth = 0;
  for (const auto &port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      continue;
    }
    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      inPortName = utils::replaceSomeChars(port.name);
      inPortWidth = port.width;

    }
    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      outPortName = utils::replaceSomeChars(port.name);
      outPortWidth = port.width;
    }
  }
  uassert(inPortWidth != 0, "Cast: input width is zero!\n");
  uassert(outPortWidth != 0, "Cast: output width is zero!\n");
  if (inPortWidth <= outPortWidth) {
    ir += "assign " + outPortName + " = " + "{{(" 
                    + std::to_string(outPortWidth - 1) + "){" + inPortName + "["
                    + std::to_string(inPortWidth - 1) + "]}}," + inPortName 
                    + "};\n";
  } else {
    ir += "assign " + outPortName + "[" + std::to_string(outPortWidth - 1) + "]"
                    + " = " + inPortName + "[" + std::to_string(inPortWidth - 1)
                    + "]" + ";\n";
    ir += "assign " + outPortName + "[" + std::to_string(outPortWidth - 2) + ":"
                    + std::to_string(0) + "]" + " = " + inPortName + "["
                    + std::to_string(outPortWidth - 2) + ":" + std::to_string(0)
                    + "]" + ";\n";
  }
  element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
  return element;
}

bool Cast::isCast(const NodeType &nodeType) {
   return nodeType.isCast();
}

} // namespace eda::hls::library::internal::verilog