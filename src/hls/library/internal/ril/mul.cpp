//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/internal/ril/mul.h"

#include "hls/library/internal/ril/element_internal_ril.h"
#include "util/string.h"

#include <cmath>

namespace eda::hls::library::internal::ril {

void Mul::estimate(const Parameters &params, Indicators &indicators) const {
  double S = params.getValue(stages);
  double Fmax = 500000.0;
  double F = Fmax * ((1 - std::exp(-S / 100.0)) + 0.1);
  double Sa = 100.0 * ((double)std::rand() / RAND_MAX) + 1;
  double W = params.getValue(width);
  double A = 100.0 * (1.0 - std::exp(50 * -(S - Sa) * (S - Sa) / 4.0)) * W;
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

SharedMetaElement Mul::create(const NodeType &nodetype,
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
    params.add(Parameter(stages, Constraint<unsigned>(1, 100), 50));

    metaElement = std::shared_ptr<MetaElement>(new Mul(lowerCaseName,
                                                       "std",
                                                       false,
                                                       params,
                                                       ports));
  return metaElement;
};

std::unique_ptr<Element> Mul::construct() const {
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  std::string inputs, outputs, ifaceWires, regs, fsm, assigns;
  std::string outputType;

  outputType = std::string("wire ");

  for (auto port : ports) {
    ifaceWires += ((port.direction == Port::IN) ? std::string("input ") 
                                                : std::string("output ")) +
        " u:" + std::to_string(port.width) + " " + replaceSomeChars(port.name) +
        ";\n";
  }

  std::string ir;
  std::vector<std::string> inPortNames;
  std::string outPortName;
  ir += "\n";
  ir += "@(*) {\n";
  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      continue;
    }
    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      inPortNames.push_back(replaceSomeChars(port.name));
    }
    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      outPortName = replaceSomeChars(port.name);
    }
  }
  ir += outPortName + " = " + inPortNames[0];
  for (size_t i = 1; i < inPortNames.size(); i++) {
    ir += " * " + inPortNames[i];
  }
  ir += ";\n}";
  element->ir = ifaceWires + ir;
  return element;
}

bool Mul::isMul(const NodeType &nodeType) {
  return nodeType.outputs.size() == 1 
      && starts_with(nodeType.name, "MUL");
}

} // namespace eda::hls::library::internal::ril