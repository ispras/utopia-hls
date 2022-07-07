//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/internal/default.h"

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

  if (name == "merge" || name == "split") {
    outputType = std::string("reg ");
  } else if (name == "add" || name == "sub" || name == "c128" ||
             name == "c181" || name == "c4" || name == "c8192" ||
             name == "mul" ||
             name == "w1" || name == "w1_add_w7" || name == "w1_sub_w7" ||
             name == "w2" || name == "w2_add_w6" || name == "w2_sub_w6" ||
             name == "w3" || name == "w3_add_w5" || name == "w3_sub_w5" ||
             name == "w5" || name == "w6"        || name == "w7" ||
             name == "shl_11" || name == "shl_8" ||
             name == "shr_14" || name == "shr_8" || name == "shr_3" ||
             name == "dup_2" || name == "clip") {
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
                        std::string("")) + port.name + ";\n";

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
      fsm += port.name;
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
      assigns += std::string("assign ") + port.name +
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
  if (name == "merge" || name == "split") {
    std::vector<std::string> portNames;
    std::string portName;
    ir += std::string("reg [31:0] state;\n");
    ir += std::string("always @(posedge clock) begin\nif (!reset) begin\n  state <= 0; end\nelse");

    for (auto port : ports) {
      if (port.name == "clock" || port.name == "reset") {
        continue;
      }
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        if (name == "merge") {
          portNames.push_back(port.name);
        } else if (name == "split") {
          portName = port.name;
        }
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        if (name == "merge") {
          portName = port.name;
        } else if (name == "split") {
          portNames.push_back(port.name);
        }
      }
    }
    unsigned counter = 0;
    for (auto currName : portNames) {
      ir += std::string(" if (state == ") + std::to_string(counter) + ") begin\n";
      ir += std::string("  state <= ") + std::to_string(++counter) + ";\n  ";
      if (name == "merge") {
        ir += portName + " <= " + currName + "; end\nelse ";
      } else if (name == "split") {
        ir += currName + " <= " + portName + "; end\nelse ";
      }
    }
    ir += std::string("begin\n  state <= 0; end\nend\n");
    element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
    //return element;
  }
  else if (name == "add" || name == "sub" || name == "mul") {
    std::vector<std::string> inPortNames;
    std::vector<std::string> outPortNames;

    for (auto port : ports) {
      if (port.name == "clock" || port.name == "reset") {
        continue;
      }
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        inPortNames.push_back(port.name);
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        outPortNames.push_back(port.name);
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
    std::string action = (name == "add" ? " + " : name == "sub" ? " - " : " * ");
    for (auto portName : inPortNames) {
      ir += (needAction ? action : "") + portName;
      if (!needAction) {
        needAction = true;
      }
    }
    ir += ";\n";
    element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
    //return element;
  }
  else if (name == "shl_11" || name == "shl_8" ||
           name == "shr_14" || name == "shr_8" || name == "shr_3" ||
           name == "clip") {
    std::string inPortName, outPortName;
    for (auto port : ports) {
      if (port.name == "clock" || port.name == "reset") {
        continue;
      }
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        inPortName = port.name;
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        outPortName = port.name;
      }
    }
    if (name == "clip") {
      ir += "assign " + outPortName + " = clip16(" + inPortName + ");\n";
      ir += std::string("function [15:0] clip16;\n") +
            "  input [15:0] in;\n" +
            "  begin\n" +
            "    if (in[15] == 1 && in[14:8] != 7'h7F)\n" +
            "      clip16 = 8'h80;\n" +
            "    else if (in[15] == 0 && in [14:8] != 0)\n" +
            "      clip16 = 8'h7F;\n" +
            "    else\n" +
            "      clip16 = in;\n" +
            "  end\n" +
            "endfunction\n";
    } else {
      ir += "assign " + outPortName + " = " + inPortName +
        (name == "shl_11" ? " << 11" :
         name == "shl_8"  ? " << 8"  :
         name == "shr_14" ? " >> 14" :
         name == "shr_8"  ? " >> 8"  :
         name == "shr_3"  ? " >> 3"  : "");
      ir += ";\n";
    }
    element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
    //return element;
  }
  else if (name == "dup_2") {
    std::string inPortName;
    std::vector<std::string> outPortNames;
    for (auto port : ports) {
      if (port.name == "clock" || port.name == "reset") {
        continue;
      }
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        inPortName = port.name;
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        outPortNames.push_back(port.name);
      }
    }
    for (auto outPortName : outPortNames) {
      ir += "assign " + outPortName + " = " + inPortName + ";\n";
    }
    element->ir = std::string("\n") + ifaceWires + inputs + outputs + ir;
    //return element;
  }
  else if (name == "c128" || name == "c181" || name == "c4" || name == "c8192" ||
           name == "w1"   || name == "w1_add_w7" || name == "w1_sub_w7" ||
           name == "w2"   || name == "w2_add_w6" || name == "w2_sub_w6" ||
           name == "w3"   || name == "w3_add_w5" || name == "w3_sub_w5" ||
           name == "w5"   || name == "w6"        || name == "w7") {
    std::string outPortName;
    for (auto port : ports) {
      if (name == "clock" || name == "reset") {
        continue;
      }
      if (!(name == "clock" || name == "reset") &&
           (port.direction == Port::OUT || port.direction == Port::INOUT)) {
        outPortName = port.name;
        break;
      }
    }
    ir += std::string("assign ") + outPortName + " = " +
          (name == "c128" ? "128" :
           name == "c181" ? "181" :
           name == "c4"   ? "4"   :
           name == "c8192" ? "8192" :
           name == "w1" ? "2841" :
           name == "w1_add_w7" ? "(2841+565)" :
           name == "w1_sub_w7" ? "(2841-565)" :
           name == "w2" ? "2676" :
           name == "w2_add_w6" ? "(2676+1108)" :
           name == "w2_sub_w6" ? "(2676-1108)" :
           name == "w3" ? "2408" :
           name == "w3_add_w5" ? "(2408+1609)" :
           name == "w3_sub_w5" ? "(2408-1609)" :
           name == "w5" ? "1609" :
           name == "w6" ? "1108" :
           name == "w7" ? "565"  :  "0") + ";\n";
    element->ir = std::string("\n") + ifaceWires + outputs + ir;
    //return element;
  } else {
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
