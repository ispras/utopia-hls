//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/internal/merge.h"
#include "util/string.h"

#include <cmath>

using namespace eda::hls::mapper;

namespace eda::hls::library {

void Merge::estimate(const Parameters &params, Indicators &indicators) const {
  unsigned inputCount = 0;
  unsigned latencySum = 0;
  unsigned widthSum = 0;

  const auto latency = params.getValue(stages);

  const auto width = 1u;

  for (const auto &port : ports) {
    widthSum += width;
    if (port.direction == Port::IN)
      inputCount++;
    else
      latencySum += latency;
  }

  double S = params.getValue(stages);
  double Fmax = 500000.0;
  double F = Fmax * (1 - std::exp(-S/20.0));
  double Sa = 100.0 * ((double)std::rand() / RAND_MAX) + 1;
  double A = 100.0 * (1.0 - std::exp(-(S - Sa) * (S - Sa) / 4.0));
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

std::shared_ptr<MetaElement> Merge::create(const NodeType &nodetype,
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

    metaElement = std::shared_ptr<MetaElement>(new Merge(lowerCaseName,
                                                         "std",
                                                         params,
                                                         ports));
  return metaElement;
};

std::shared_ptr<MetaElement> Merge::createDefaultElement() {
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

  ports.push_back(Port("z1",
                      Port::IN,
                      16,
                      model::Parameter(std::string("width"), 16)));

  ports.push_back(Port("z2",
                       Port::IN,
                       16,
                       model::Parameter(std::string("width"), 16)));

  ports.push_back(Port("z",
                      Port::OUT,
                      16,
                      model::Parameter(std::string("width"), 16)));
  Parameters params;
  params.add(Parameter(stages, Constraint<unsigned>(1, 100), 10));

  metaElement = std::shared_ptr<MetaElement>(new Merge("merge",
                                                       "std",
                                                       params,
                                                       ports));
  return metaElement;
};


std::unique_ptr<Element> Merge::construct(const Parameters &params) const {
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
      (port.width > 1 ? std::string("[") + std::to_string(port.width - 1) + ":0] " :
                        std::string("")) + replaceSomeChars(port.name) + ";\n";

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
  std::vector<std::string> portNames;
  std::string portName;
  ir += std::string("reg [31:0] state;\n");
  ir += std::string("always @(posedge clock) begin\nif (!reset) begin\n  state <= 0; end\nelse");

  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      continue;
    }
    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      portNames.push_back(replaceSomeChars(port.name));
    }
    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      portName = replaceSomeChars(port.name);
    }
  }
  unsigned counter = 0;
  for (auto currName : portNames) {
    ir += std::string(" if (state == ") + std::to_string(counter) + ") begin\n";
    ir += std::string("  state <= ") + std::to_string(++counter) + ";\n  ";
    ir += portName + " <= " + currName + "; end\nelse ";
  }
  ir += std::string("begin\n  state <= 0; end\nend\n");
  element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
  return element;
}
} // namespace eda::hls::library
