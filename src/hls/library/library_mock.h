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
  std::string inputs, outputs, iface_wires, regs, fsm, assigns;
  unsigned int pos = 0, input_length = 0, output_length = 0;

  for (auto port : ports) {
    if (port.name != "clock" && port.name != "reset") {
      if (port.direction == Port::IN || port.direction == Port::INOUT) {
        input_length += port.width;
      }
      if (port.direction == Port::OUT || port.direction == Port::INOUT) {
        output_length += port.width;
      }
    }
  }

  if (output_length == 0) {
    element->ir = iface_wires;
    return element;
  }

  // FSM can be create iff there is at least one input in the design.
  bool fsmNotCreated = true;

  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      iface_wires += std::string("input ") + port.name + ";\n";
      continue;
    }

    std::string portDeclr =
      (port.width > 1 ? std::string("[") + std::to_string(port.width - 1) + ":0]" :
                        std::string("")) + port.name + ";\n";

    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      if (port.direction == Port::IN) {
        iface_wires += std::string("input ") + portDeclr;
      } else {
        iface_wires += std::string("inout ") + portDeclr;
      }
      inputs += std::string("wire ") + portDeclr;

      // Create the first stage of pipeline.
      if (fsmNotCreated) {
        regs += std::string("reg [") + std::to_string(input_length - 1) + ":0] state_0;\n";
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
        iface_wires += std::string("output ") + portDeclr;
      }
      outputs += std::string("wire ") + portDeclr;
      if (input_length < output_length && pos != 0) {
        pos -= port.width; // FIXME
      }
      assigns += std::string("assign ") + port.name +
                 " = state_" +
                 (port.latency == 0 ? "0" : std::to_string(port.latency - 1)) +
                 "[" + std::to_string((pos + port.width - 1) % output_length) +
                 ":" + std::to_string(pos % output_length) + "];\n";
      pos += port.width;
    }
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
    regs += std::string("reg [") + std::to_string(input_length - 1) + ":0] state_" + std::to_string(i) + ";\n";
    if (input_length > 2) {
      fsm += std::string("state_") + std::to_string(i) +
                          " = {state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 2) + ":0], " +
                          "state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 1) + "]};\n";
    } else if (input_length == 2) {
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

  element->ir = std::string("\n") + iface_wires + inputs + outputs + regs + fsm + assigns;
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
