//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/internal/default.h"
#include "util/string.h"

#include <cmath>

using namespace eda::hls::mapper;

namespace eda::hls::library {

void Default::estimate(
    const Parameters &params, Indicators &indicators) const {
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
//  double Areg = 1.0;
//  double Apipe = S * widthSum * Areg;
  double Fmax = 500000.0;
  double F = Fmax * (1 - std::exp(-S/20.0));
//  double C = inputCount * latencySum;
//  double N = (C == 0 ? 0 : C * std::log((Fmax / (Fmax - F)) * ((C - 1) / C)));
//  double A = C * std::sqrt(N) + Apipe;
  double Sa = 100.0 * ((double)std::rand() / RAND_MAX) + 1;
  double A = 100.0 * (1.0 - std::exp(-(S - Sa) * (S - Sa) / 4.0));
  double P = A;
  double D = 1000000000.0 / F;

  indicators.ticks = static_cast<unsigned>(S);
  indicators.power = static_cast<unsigned>(P);
  indicators.area  = static_cast<unsigned>(A);
  indicators.delay = static_cast<unsigned>(D);
/*
  std::cout << "Node: " << name << std::endl;
  std::cout << "ticks: " << indicators.ticks << " delay: " << indicators.delay;
  std::cout << " freq: " << F << std::endl;
*/
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

std::shared_ptr<MetaElement> Default::create(const NodeType &nodetype,
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

    metaElement = std::shared_ptr<MetaElement>(new Default(lowerCaseName,
                                                           "std",
                                                           params,
                                                           ports));
  return metaElement;
};


std::unique_ptr<Element> Default::construct(
    const Parameters &params) const {
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  std::string inputs, outputs, ifaceWires, regs, fsm, assigns;
  unsigned pos = 0, inputLength = 0, outputLength = 0;
  std::string outputType;
  bool quickProcess = true;

  const auto latency = 1u;

  if (libName == "merge" || libName == "split") {
    outputType = std::string("reg ");
  } else if (libName == "add"    || libName == "sub"   || libName == "mul"   ||
             libName == "shl11"  || libName == "shl8"  || libName == "shr14" || 
             libName == "shr8"   || libName == "shr3"  || libName == "clip"  || 
             libName == "lt"     || libName == "gt"    || libName == "eq"    ||
             libName == "ne"     || libName == "le"    || libName == "ge") {
    outputType = std::string("wire ");
  } else {
    outputType = std::string("wire ");
    quickProcess = false;
  }

  if (!quickProcess) {
    for (auto port : ports) {
      if (port.name != "clock" && port.name != "reset") {
        if (port.direction == Port::IN || port.direction == Port::INOUT) {
          inputLength += port.width;
        }
        if (port.direction == Port::OUT || port.direction == Port::INOUT) {
          outputLength += port.width;
        }
      }
    }
  }

  // FSM can be create iff there is at least one input in the design.
  bool fsmNotCreated = true;

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
      if (quickProcess) {
        continue;
      }

      // Create the first stage of pipeline.
      if (fsmNotCreated) {
        regs += std::string("reg [") + std::to_string(inputLength - 1) + ":0] state_0;\n";
        fsm += std::string("always @(posedge clock) begin\n");
        fsm += std::string("state_0 <= {");
        fsmNotCreated = false;
      } else {
        fsm += std::string(", ");
      }
      fsm += replaceSomeChars(port.name);
    }

    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      if (port.direction == Port::OUT) {
        ifaceWires += std::string("output ") + portDeclr;
      }
      outputs += outputType + portDeclr;
      if (quickProcess) {
        continue;
      }

      if (inputLength < outputLength && pos != 0) {
        pos -= port.width; // FIXME
      }
      uassert(outputLength != 0, "All the outputs have zero width!");
      assigns += std::string("assign ") + replaceSomeChars(port.name) +
                 " = state_" +
                 (latency == 0 ? "0" : std::to_string(latency - 1)) +
                 "[" + std::to_string((pos + port.width - 1) % outputLength) +
                 ":" + std::to_string(pos % outputLength) + "];\n";
      pos += port.width;
    }
  }

  if (!quickProcess && outputLength == 0) {
    element->ir = std::string("\n") + ifaceWires + inputs;
  }

  std::string ir;
  if (libName == "add" || libName == "sub" || libName == "mul" ||
           libName == "lt"  || libName ==  "gt" || libName == "eq"  ||
           libName == "ne"  || libName ==  "le" || libName == "ge") {
    std::vector<std::string> inPortNames;
    std::vector<std::string> outPortNames;

    for (auto port : ports) {
      if (port.name == "clock" || port.name == "reset") {
        continue;
      }
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        inPortNames.push_back(replaceSomeChars(port.name));
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        outPortNames.push_back(replaceSomeChars(port.name));
      }
    }
    if (libName == "lt" || libName == "gt" || libName == "eq" ||
        libName == "ne" || libName == "le" || libName == "ge") {
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
    std::string action;
    if (libName == "add") {
      action = " + ";
    } else if (libName == "sub") {
      action = " - ";
    } else if (libName == "mul") {
      action = " * ";
    } else if (libName == "lt") {
      action = " < ";
    } else if (libName == "gt") {
      action = " > ";
    } else if (libName == "eq") {
      action = " == ";
    } else if (libName == "ne") {
      action = " != ";
    } else if (libName == "le") {
      action = " <= ";
    } else {
      action = " >= ";
    } 
    for (auto portName : inPortNames) {
      ir += (needAction ? action : "") + "$signed(" + portName + ")";
      if (!needAction) {
        needAction = true;
      }
    }
    ir += ";\n";
    element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
    //return element;
  }
  else if (libName == "shl11" || libName == "shl8" || libName == "shr14" ||
           libName == "shr8"  || libName == "shr3") {
    std::string inPortName, outPortName;
    for (auto port : ports) {
      if (port.name == "clock" || port.name == "reset") {
        continue;
      }
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        inPortName = replaceSomeChars(port.name);
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        outPortName = replaceSomeChars(port.name);
      }
    }

    ir += "assign " + outPortName + " = " + "$signed(" + inPortName + ")" +
      (libName == "shl11" ? " <<< 11" :
        libName == "shl8"  ? " <<< 8"  :
        libName == "shr14" ? " >>> 14" :
        libName == "shr8"  ? " >>> 8"  :
        libName == "shr3"  ? " >>> 3"  : "");
    ir += ";\n";
    element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
    //return element;
  }
  // TODO: discuss conventions for sequence of parameters.
  else {
  // Finish creating the first stage of pipeline.
    if (!fsmNotCreated) {
      fsm += std::string("};\n");
    } else {
      regs += std::string("reg [") + std::to_string(pos - 1) + ":0] state_0;\n";
    }

    // Extract latency and construct a cascade of assignments.
    // Indicators indicators;
    // estimate(params, indicators);
    // indicators.latency; FIXME

    for (unsigned i = 1; (i < latency) && !fsmNotCreated; i++) {
      regs += std::string("reg [") + std::to_string(inputLength - 1) + ":0] state_" + std::to_string(i) + ";\n";
      if (inputLength > 2) {
        fsm += std::string("state_") + std::to_string(i) +
                            " <= {state_" + std::to_string(i - 1) + "[" + std::to_string(inputLength - 2) + ":0], " +
                            "state_" + std::to_string(i - 1) + "[" + std::to_string(inputLength - 1) + "]};\n";
      } else if (inputLength == 2) {
        fsm += std::string("state_") + std::to_string(i) + "[1:0] <= {state_" + std::to_string(i - 1) +
                           "[0], state_" + std::to_string(i - 1) + "[1]};\n";
      }
      else {
        fsm += std::string("state_") + std::to_string(i) + "[0] <= state_" + std::to_string(i - 1) + "[0];\n";
      }
    }
    if (!fsmNotCreated) {
      fsm += std::string("end\n");
    }

    element->ir = std::string("\n") + ifaceWires + inputs + outputs + regs + fsm + assigns;
  }
  return element;
}

} // namespace eda::hls::library
