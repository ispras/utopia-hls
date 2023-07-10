//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/verilog/dup.h"

#include "utils/string.h"

#include <cmath>

namespace eda::hls::library::internal::verilog {

void Dup::estimate(const Parameters &params, Indicators &indicators) const {
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

std::shared_ptr<MetaElement> Dup::create(const NodeType &nodetype,
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
    params.add(Parameter(stages, Constraint<unsigned>(1, 100), 0));
    params.add(Parameter(width, Constraint<unsigned>(1, 128), 16));

    metaElement = std::shared_ptr<MetaElement>(new Dup(lowerCaseName,
                                                       "std",
                                                       true,
                                                       params,
                                                       ports));
  return metaElement;
};


std::shared_ptr<MetaElement> Dup::createDefaultElement() {
  std::shared_ptr<MetaElement> metaElement;
  std::vector<Port> ports;

  ports.push_back(Port("clock",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1)));

  ports.push_back(Port("reset",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1)));

  ports.push_back(Port("x",
                      Port::IN,
                      16,
                      model::Parameter(std::string("width"), 16)));

  ports.push_back(Port("y",
                       Port::OUT,
                       16,
                       model::Parameter(std::string("width"), 16)));

  ports.push_back(Port("z",
                      Port::OUT,
                      16,
                      model::Parameter(std::string("width"), 16)));
  Parameters params;
  params.add(Parameter(stages, Constraint<unsigned>(1, 100), 10));

  metaElement = std::shared_ptr<MetaElement>(new Dup("dup_2",
                                                     "std",
                                                     true,
                                                     params,
                                                     ports));
  return metaElement;
};


std::unique_ptr<Element> Dup::construct() const {
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  std::string inputs, outputs, ifaceWires, regs, fsm, assigns;
  std::string outputType;

  outputType = std::string("wire ");

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
  std::string inPortName;
  std::vector<std::string> outPortNames;
  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      continue;
    }
    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      inPortName = utils::replaceSomeChars(port.name);
    }
    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      outPortNames.push_back(utils::replaceSomeChars(port.name));
    }
  }
  for (auto outPortName : outPortNames) {
    ir += "assign " + outPortName + " = " + inPortName + ";\n";
  }
  element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
  return element;
}

bool Dup::isDup(const NodeType &nodeType) {
   return nodeType.isDup();
}

} // namespace eda::hls::library::internal::verilog