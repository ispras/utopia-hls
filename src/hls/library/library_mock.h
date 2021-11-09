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
  bool quickProcess = false;

  if (name == "merge" || name == "split" || name == "delay") {
    outputType = std::string("reg ");
    quickProcess = true;
  } else {
    outputType = std::string("wire ");
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


  // Finish creating the first stage of pipeline.
  if (!fsmNotCreated) {
    fsm += std::string("};\n");
  } else {
    regs += std::string("reg [") + std::to_string(pos - 1) + ":0] state_0;\n";
  }

  // Extract frequency.
  // FIXME
  // unsigned f = params.value("f");
  unsigned f = 6;
  for (unsigned i = 1; (i < f) && !fsmNotCreated; i++) {
    regs += std::string("reg [") + std::to_string(inputLength - 1) + ":0] state_" + std::to_string(i) + ";\n";
    if (inputLength > 2) {
      fsm += std::string("state_") + std::to_string(i) +
                          " = {state_" + std::to_string(i - 1) + "[" + std::to_string(inputLength - 2) + ":0], " +
                          "state_" + std::to_string(i - 1) + "[" + std::to_string(inputLength - 1) + "]};\n";
    } else if (inputLength == 2) {
      fsm += std::string("state_") + std::to_string(i) + "[1:0] = {state_" + std::to_string(i - 1) +
                         "[0], state_" + std::to_string(i - 1) + "[1]};\n";
    }
    else {
      fsm += std::string("state_") + std::to_string(i) + "[0] = state_" + std::to_string(i - 1) + "[0];\n";
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
  } else*/ if (name == "add" || name == "sub") {
    ports.push_back(Port("a", Port::IN, 0, 4));
    ports.push_back(Port("b", Port::IN, 0, 4));
    ports.push_back(Port("c", Port::OUT, 2, 4));
    ports.push_back(Port("d", Port::OUT, 2, 1));
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

  return std::shared_ptr<MetaElement>(new MetaElementMock(nodetype.name, params, ports));
}

static struct LibraryInitializer {
  LibraryInitializer() {
    /*Library::get().add(MetaElementMock::create("merge"));
    Library::get().add(MetaElementMock::create("split"));
    Library::get().add(MetaElementMock::create("delay"));*/
    Library::get().add(MetaElementMock::create("add"));
    Library::get().add(MetaElementMock::create("sub"));
  }
} libraryInitializer;

} // namespace eda::hls::library
