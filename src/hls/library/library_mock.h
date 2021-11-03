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
  std::string inputs, outputs, iface_wires;
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

  bool first_port = true;
  for (auto port : ports) {
    if (port.name == "clock" || port.name == "reset") {
      iface_wires += std::string("input ") + port.name + ";\n";
      continue;
    }

    if (port.direction == Port::IN || port.direction == Port::INOUT) {
      if (port.direction == Port::IN) {
        iface_wires += std::string("input [") + std::to_string(port.width - 1) + ":0] " + port.name + ";\n";
      } else {
        iface_wires += std::string("inout [") + std::to_string(port.width - 1) + ":0] " + port.name + ";\n";
      }
      if (first_port) {
        inputs += std::string("reg [") + std::to_string(input_length - 1) + ":0] stage_0 = {";
        first_port = false;
      } else {
        inputs += std::string(", ");
      }
      inputs += port.name;
    }

    if (port.direction == Port::OUT || port.direction == Port::INOUT) {
      iface_wires += std::string("output [") + std::to_string(port.width - 1) + ":0] " + port.name + ";\n";
      outputs += std::string("wire [") + std::to_string(port.width - 1) + ":0] " + port.name +
                 " = stage_" + std::to_string(port.latency - 1) +
                 "[" + std::to_string((pos + port.width - 1) % output_length) + ":" +
                       std::to_string(pos % output_length) + "];\n";
      pos += port.width;
    }
  }

  inputs += std::string("};\n");

  // Extract frequency.
  // FIXME
  // unsigned f = params.value("f");
  unsigned f = 6;
  for (unsigned i = 1; i < f; i++) {
    if (input_length > 2) {
      inputs += std::string("reg [") + std::to_string(input_length - 1) + ":0] state_" + std::to_string(i) +
                            " = {state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 2) + ":0], " +
                            "state_" + std::to_string(i - 1) + "[" + std::to_string(input_length - 1) + "]};\n";
    } else {
      inputs += std::string("state_") + std::to_string(i) + "[1:0] = state_" + std::to_string(i - 1) + "[0:1];\n";
    }
  }

  element->ir = iface_wires + inputs + outputs;
  return element;
}

inline void MetaElementMock::estimate(
    const Parameters &params, Indicators &indicators) const {
  // TODO: estimations of l, f, p, a
  indicators.frequency = params.value("f");
  indicators.throughput = indicators.frequency;
  indicators.latency = 1000000;
  indicators.power = 5;
  indicators.area = 10000;
}

inline std::shared_ptr<MetaElement> MetaElementMock::create(
    const std::string &name) {
  Parameters params("");
  params.add(Parameter("f", Constraint(1, 1000), 100));

  /// Populate ports for different library elements
  Ports ports;
  ports.push_back(Port("clock", Port::IN, 0, 1));
  ports.push_back(Port("reset", Port::IN, 0, 1));

  if (name == "merge") {
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
  } else if (name == "add" || name == "sub") {
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

  /// Copy ports from model
  Ports ports;
  for (const auto *input: nodetype.inputs) {
    ports.push_back(Port(input->name, Port::IN, input->latency, 1 /*arg->length*/));
  }
  for (const auto *output: nodetype.outputs) {
    ports.push_back(Port(output->name, Port::OUT, output->latency, 1 /*arg->length*/));
  }

  return std::shared_ptr<MetaElement>(new MetaElementMock(nodetype.name, params, ports));
}

} // namespace eda::hls::library
