//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "util/assert.h"

#include <cmath>

using namespace eda::hls::model;

namespace eda::hls::library {

struct MetaElementMock final : public MetaElement {
  MetaElementMock(const std::string &name,
                  const Parameters &params,
                  const Ports &ports):
      MetaElement(name, params, ports) {}

  virtual std::unique_ptr<Element> construct(
      const Parameters &params) const override;
  virtual void estimate(
      const Parameters &params, Indicators &indicators) const override;

  static std::shared_ptr<MetaElement> create(const std::string &name);
  static std::shared_ptr<MetaElement> create(const NodeType &nodetype);
};

inline std::unique_ptr<Element> MetaElementMock::construct(
    const Parameters &params) const {
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  std::string inputs, outputs, ifaceWires, regs, fsm, assigns;
  unsigned pos = 0, inputLength = 0, outputLength = 0;
  std::string outputType;
  bool quickProcess = true;

  if (name == "merge" || name == "split" || name == "delay") {
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
      (port.width > 1 ? std::string("[") + std::to_string(port.width - 1) + ":0]" :
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
                 (port.latency == 0 ? "0" : std::to_string(port.latency - 1)) +
                 "[" + std::to_string((pos + port.width - 1) % outputLength) +
                 ":" + std::to_string(pos % outputLength) + "];\n";
      pos += port.width;
    }
  }

  if (!quickProcess && outputLength == 0) {
    element->ir = std::string("\n") + ifaceWires + inputs;
    return element;
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
    return element;
  }
  else if (name == "delay") {
    std::string inPort, outPort;
    unsigned d = 3; // FIXME
    regs += std::string("reg [31:0] state;\n");

    for (auto port : ports) {
      if (port.name == "clock" || port.name == "reset") {
        continue;
      }
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        inPort = port.name;
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        outPort = port.name;
      }
    }

    ir += std::string(" if (state == 0) begin\n  state <= 1;\n  s0 <= ") + inPort + "; end\nelse";
    regs += std::string("reg [31:0] s0;\n");
    for (unsigned i = 1; i < d; i++) {
      regs += std::string("reg [31:0] s") + std::to_string(i) + ";\n";
      ir += std::string(" if (state == ") + std::to_string(i) + ") begin\n";
      ir += std::string("  state <= ") + std::to_string(i + 1) + ";\n";
      ir += std::string("  s") + std::to_string(i) + " <= s" + std::to_string(i - 1) + "; end\nelse";
    }
    ir += std::string(" begin\n  state <= 0;\n  ") + outPort + " <= s" +
          std::to_string(d - 1) + "; end\nend\n";
    regs += std::string("always @(posedge clock) begin\nif (!reset) begin\n  state <= 0; end\nelse");
    element->ir = std::string("\n") + ifaceWires + inputs + outputs + regs + ir;
    return element;
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
    return element;
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
    return element;
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
    return element;
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
      else if (port.direction == Port::OUT || port.direction == Port::INOUT) {
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
    return element;
  }

  // Finish creating the first stage of pipeline.
  if (!fsmNotCreated) {
    fsm += std::string("};\n");
  } else {
    regs += std::string("reg [") + std::to_string(pos - 1) + ":0] state_0;\n";
  }

  // Extract latency and construct a cascade of assignments.
  // Indicators indicators;
  // estimate(params, indicators);
  unsigned latency = 3; // indicators.latency; FIXME

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
  return element;
}

inline void MetaElementMock::estimate(
    const Parameters &params, Indicators &indicators) const {
  unsigned inputCount = 0;
  unsigned latencySum = 0;

  for (const auto &port : ports) {
    if (port.direction == Port::IN)
      inputCount++;
    else
      latencySum += port.latency;
  }

  double F = params.value("f");
  double Fmax = 1000000.0;
  double C = inputCount * latencySum;
  double N = (C == 0 ? 0 : C * std::log((Fmax / (Fmax - F)) * ((C - 1) / C)));
  double A = C * std::sqrt(N);
  double P = A;

  indicators.frequency  = static_cast<unsigned>(F);
  indicators.throughput = static_cast<unsigned>(F);
  indicators.latency    = static_cast<unsigned>(N);
  indicators.power      = static_cast<unsigned>(P);
  indicators.area       = static_cast<unsigned>(A);
}

inline std::shared_ptr<MetaElement> MetaElementMock::create(
    const std::string &name) {
  Parameters params("");
  params.add(Parameter("f", Constraint(1, 1000), 100));

  // Populate ports for different library elements
  Ports ports;
  ports.push_back(Port("clock", Port::IN, 0, 1));
  ports.push_back(Port("reset", Port::IN, 0, 1));

  /*if (name == "merge") {
    ports.push_back(Port("in1", Port::IN, 0, 1));
    ports.push_back(Port("in2", Port::IN, 0, 1));
    ports.push_back(Port("out", Port::OUT, 1, 1));
  } else if (name == "split") {
    ports.push_back(Port("in", Port::IN, 0, 1));
    ports.push_back(Port("out1", Port::OUT, 1, 1));
    ports.push_back(Port("out2", Port::OUT, 1, 1));
  } else if (name == "delay") {
    ports.push_back(Port("in", Port::IN, 0, 1));
    ports.push_back(Port("out", Port::OUT, 1, 1));
  } else*/ if (name == "add1" || name == "sub1") {
    ports.push_back(Port("a", Port::IN, 0, 4));
    ports.push_back(Port("b", Port::IN, 0, 4));
    ports.push_back(Port("d", Port::OUT, 2, 1));
    ports.push_back(Port("c", Port::OUT, 2, 4));
  } else {
    uassert(false, "Call createMetaElement by NodeType for element " << name);
  }

  return std::shared_ptr<MetaElement>(new MetaElementMock(name, params, ports));
}

inline std::shared_ptr<MetaElement> MetaElementMock::create(
    const NodeType &nodetype) {
  Parameters params("");
  params.add(Parameter("f", Constraint(1, 1000), 100));

  // Copy ports from model
  Ports ports;
  for (const auto *input: nodetype.inputs) {
    ports.push_back(Port(input->name, Port::IN, input->latency, 1 /*arg->length*/));
  }
  for (const auto *output: nodetype.outputs) {
    ports.push_back(Port(output->name, Port::OUT, output->latency, 1 /*arg->length*/));
  }

  // Add clk and rst ports: these ports are absent in the lists above.
  ports.push_back(Port("clock", Port::IN, 0, 1));
  ports.push_back(Port("reset", Port::IN, 0, 1));

  std::string lowerCaseName = nodetype.name;
  unsigned i = 0;
  while (lowerCaseName[i]) {
    lowerCaseName[i] = tolower(lowerCaseName[i]);
    i++;
  }

  return std::shared_ptr<MetaElement>(new MetaElementMock(lowerCaseName, params, ports));
}

// MetaElement:
//   Name of the element
//   List of its Parameter(s)
//   getIndicators function
//   Generation facilities: function, correspondent file
//   (List of) File(s)
//   (Module in FIRRTL)
struct PreLibraryElement {
  const std::string name;
  const Parameters parameters;
  const Ports ports;
  Indicators indicators;
  void (*getIndicators)(const Parameters&, const Ports&, Indicators&);
  std::string generator;
  std::string fileName;

  PreLibraryElement(const std::string &name, const Parameters &parameters,
    const Ports &ports, void (*getIndicators)(const Parameters&, const Ports&, Indicators&)) :
    name(name), parameters(parameters), ports(ports), getIndicators(getIndicators) {
    getIndicators(parameters, ports, indicators);
  };

  static void indicatorsTest(const Parameters &params, const Ports& ports, Indicators &indicators) {
    unsigned inputCount = 0;
    unsigned latencySum = 0;

    for (const auto &port : ports) {
      if (port.direction == Port::IN)
        inputCount++;
      else
        latencySum += port.latency;
    }

    double F = params.value("f");
    double Fmax = 1000000.0;
    double C = inputCount * latencySum;
    double N = (C == 0 ? 0 : C * std::log((Fmax / (Fmax - F)) * ((C - 1) / C)));
    double A = C * std::sqrt(N);
    double P = A;

    indicators.frequency  = static_cast<unsigned>(F);
    indicators.throughput = static_cast<unsigned>(F);
    indicators.latency    = static_cast<unsigned>(N);
    indicators.power      = static_cast<unsigned>(P);
    indicators.area       = static_cast<unsigned>(A);
  };
};

static struct PreLibraryElements {
  static PreLibraryElement getAdd(Parameters &params, Ports &ports) {
    PreLibraryElement element("add", params, ports, PreLibraryElement::indicatorsTest);
    element.generator = std::string("example: flopoco -opt $P1 $OUT");
    element.fileName  = std::string("verilog/add.v");
    return element;
  }
  static PreLibraryElement getSub(Parameters &params, Ports &ports) {
    PreLibraryElement element("sub", params, ports, PreLibraryElement::indicatorsTest);
    element.generator = std::string("example: flopoco -opt $P1 $OUT");
    element.fileName  = std::string("verilog/sub.v");
    return element;
  }

  PreLibraryElements() {
    Parameters parameters("");
    parameters.add(Parameter("f", Constraint(1, 1000), 100));

    Port porta("a", Port::IN, 0, 4), portb("b", Port::IN, 0, 4);
    Port portc("c", Port::OUT, 1, 4);
    Ports ports;
    ports.push_back(porta);
    ports.push_back(portb);
    ports.push_back(portc);

    preLibraryElements.push_back(getAdd(parameters, ports));
    preLibraryElements.push_back(getSub(parameters, ports));
  }

  std::vector<PreLibraryElement> preLibraryElements;
} preLibraryElements;

static struct LibraryInitializer {
  LibraryInitializer() {
    /*Library::get().add(MetaElementMock::create("merge"));
    Library::get().add(MetaElementMock::create("split"));
    Library::get().add(MetaElementMock::create("delay"));*/
    Library::get().add(MetaElementMock::create("add1"));
    Library::get().add(MetaElementMock::create("sub1"));
  }
} libraryInitializer;

} // namespace eda::hls::library
