//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/verilog/add.h"

#include "util/string.h"

#include <cmath>

namespace eda::hls::library::internal::verilog {

void Add::estimate(const Parameters &params, Indicators &indicators) const {
  double S = params.getValue(stages);
  double Fmax = 500000.0;
  double F = Fmax * ((1 - std::exp(-S / 40.0)) + 0.1);
  double Sa = 100.0 * ((double)std::rand() / RAND_MAX) + 1;
  double W = params.getValue(width);
  double A = 100.0 * (1.0 - std::exp(10 * -(S - Sa) * (S - Sa) / 4.0)) * W;
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

std::shared_ptr<MetaElement> Add::create(const NodeType &nodetype,
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
    params.add(Parameter(stages, Constraint<unsigned>(1, 100), 10));
    params.add(Parameter(width, Constraint<unsigned>(1, 128), 16));

    metaElement = std::shared_ptr<MetaElement>(new Add(lowerCaseName,
                                                       "std",
                                                       false,
                                                       params,
                                                       ports));
  return metaElement;
};

std::unique_ptr<Element> Add::construct() const {
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

  std::vector<std::string> inPortNames;
  std::vector<std::string> outPortNames;
  std::string ir;

  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      continue;
    }
    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      inPortNames.push_back(utils::replaceSomeChars(port.name));
    }
    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      outPortNames.push_back(utils::replaceSomeChars(port.name));
    }
  }
  ir += "assign {";
  bool comma = false;
  for (auto portName : outPortNames) {
    ir += (comma ? ", " : "") + portName;
    if (!comma) {
      comma = true;
    }
  }
  ir += "} = ";
  bool needAction = false;
  std::string action = " + ";
  for (auto portName : inPortNames) {
    ir += (needAction ? action : "") + "$signed(" + portName + ")";
    if (!needAction) {
      needAction = true;
    }
  }
  ir += ";\n";
  element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
  return element;
}

bool Add::isAdd(const NodeType &nodeType) {
  return nodeType.outputs.size() == 1 
      && utils::starts_with(nodeType.name, "ADD");
}

} // namespace eda::hls::library::internal::verilog