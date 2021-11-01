//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include <iostream>

namespace eda::hls::library {

struct Constraint final {
  Constraint(unsigned loValue, unsigned hiValue):
    loValue(loValue), hiValue(hiValue) {}
  Constraint(const Constraint &c): loValue(c.loValue), hiValue(c.hiValue) {}
  const unsigned loValue;
  const unsigned hiValue;
};

struct Parameter final {
  Parameter(const std::string &name, const Constraint &constraint):
    name(name), constraint(constraint) {};
  Parameter(const Parameter &p): name(p.name), constraint(p.constraint) {}
  const std::string name;
  const Constraint constraint;
};

typedef std::vector<Parameter> Parameters;

/// Description of a class of modules with the given names,
///  parameter list, and allowed ranges of their values.
struct MetaElementDescriptor final {
  MetaElementDescriptor(const std::string &name, const Parameters &parameters):
    name(name), parameters(parameters) {}
  MetaElementDescriptor(const MetaElementDescriptor &med):
    name(med.name), parameters(med.parameters) {}

  const std::string name;

  /// List of parameters with ranges on their values.
  const Parameters parameters;
};

/// RTL port with name, direction, and width.
struct Port {
  enum Direction { IN, OUT, INOUT };

  Port(const std::string &name, const Direction &direction,
       unsigned latency, unsigned width):
    name(name), direction(direction), latency(latency), width(width) {}
  Port(const Port &port): name(port.name), direction(port.direction),
    latency(port.latency), width(port.width) {}

  const std::string name;
  const Direction direction;
  const unsigned latency;
  const unsigned width;
};

typedef std::vector<Port> Ports;

/// Description of a module with the given name and values of parameters.
struct ElementDescriptor final {
  explicit ElementDescriptor(const Ports &ports): ports(ports) {}

  // TODO add mutual relation between spec ports and impl ports
  const Ports ports;

  // TODO there should be different IRs: MLIR FIRRTL or Verilog|VHDL described in FIRRTL
  std::string ir;
};

struct ElementCharacteristics final {
  ElementCharacteristics(
    unsigned frequency, unsigned throughput, unsigned power, unsigned area):
    frequency(frequency), throughput(throughput), power(power), area(area) {}

  const unsigned frequency;
  const unsigned throughput;
  const unsigned power;
  const unsigned area;
};

struct ElementArguments final {
  explicit ElementArguments(const std::string &name): name(name) {}

  const std::string name;
  std::map<const std::string, unsigned> args;
};

class Library final {
public:
  Library();

  // Return a list of parameters for the module with the given name
  // (and correspodent functionality).
  const MetaElementDescriptor& find(const std::string &name) const;

  // Return a module with the selected set of parameters
  // (where f is an additional parameter).
  std::unique_ptr<ElementDescriptor> construct(const ElementArguments &args) const;

  // Return characteristics for the selected set of parameters.
  std::unique_ptr<ElementCharacteristics> estimate(const ElementArguments &args) const;

private:
  std::vector<MetaElementDescriptor> library;
};

struct VerilogNodeTypePrinter final {
  VerilogNodeTypePrinter(const eda::hls::model::NodeType &type, const Library &library):
    type(type), library(library) {}
  void print(std::ostream &out) const;

  const eda::hls::model::NodeType &type;
  const Library &library;
};

struct VerilogGraphPrinter final {
  VerilogGraphPrinter(const eda::hls::model::Graph &graph, const Library &library):
    graph(graph), library(library) {}

  void printChan(std::ostream &out, const eda::hls::model::Chan &chan) const;
  void print(std::ostream &out) const;

  const eda::hls::model::Graph &graph;
  const Library &library;
};

std::ostream& operator <<(std::ostream &out, const VerilogNodeTypePrinter &printer);
std::ostream& operator <<(std::ostream &out, const VerilogGraphPrinter &printer);

} // namespace eda::hls::library
