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

/// RTL port with name, direction, and width.
struct Port {
  enum Direction { IN, OUT, INOUT };

  Port(const std::string &name, const Direction &direction, unsigned width):
    name(name), direction(direction), width(width) {}
  Port(const Port &port) :
    name(port.name), direction(port.direction), width(port.width) {}

  const std::string name;
  const Direction direction;
  const unsigned width;
};

struct Constraint final {
  Constraint(unsigned loValue, unsigned hiValue):
    loValue(loValue), hiValue(hiValue) {}
  Constraint(const Constraint &c): loValue(c.loValue), hiValue(c.hiValue) {}
  const unsigned loValue;
  const unsigned hiValue;
};

struct Parameter final {
  Parameter(const Port &port, const Constraint &constraint):
    port(port), constraint(constraint) {};
  Parameter(const Parameter &p): port(p.port), constraint(p.constraint) {}
  const Port port;
  const Constraint constraint;
};

typedef std::vector<Port> ElementArguments;
typedef std::vector<Parameter> Parameters;

/// Description of a class of modules with the given names,
///  port list, and allowed ranges of their values.
struct MetaElementDescriptor final {
  MetaElementDescriptor(const std::string &name, const Parameters &parameters):
    name(name), parameters(parameters) {}
  MetaElementDescriptor(const MetaElementDescriptor &med):
    name(med.name), parameters(med.parameters) {}

  const std::string name;

  /// List of parameters with ranges on their values.
  const Parameters parameters;
};

/// Description of a module with the given name and values of parameters.
struct ElementDescriptor final {
  explicit ElementDescriptor(const ElementArguments &args): ports(args) {}

  // TODO add mutual relation between spec ports and impl ports
  const ElementArguments ports;

  // TODO there should be different IRs: MLIR FIRRTL or Verilog|VHDL described in FIRRTL
  std::string ir;
};

struct ExtendedPort final : Port {
  ExtendedPort(const std::string &name, const Direction &direction,
               unsigned width, unsigned latency):
    Port(name, direction, width), latency(latency) {}

  ExtendedPort(const Port &port, unsigned latency):
    Port(port), latency(latency) {}

  ExtendedPort(const ExtendedPort &ep):
    Port(ep.name, ep.direction, ep.width), latency(ep.latency) {}

  const unsigned latency;
};

typedef std::vector<ExtendedPort> ExtendedElementArguments;

struct ElementCharacteristics final {
  ElementCharacteristics(const ExtendedElementArguments &latencies,
    unsigned frequency, unsigned throughput, unsigned power, unsigned area):
    latencies(latencies), frequency(frequency), throughput(throughput), power(power), area(area) {}

  const ExtendedElementArguments latencies;
  const unsigned frequency;
  const unsigned throughput;
  const unsigned power;
  const unsigned area;
};

class Library final {
public:
  Library();

  // Return a list of parameters for the module with the given name
  // (and correspodent functionality).
  const MetaElementDescriptor& find(const std::string &name) const;

  // Return a module with the selected set of parameters
  // (where f is an additional parameter).
  std::unique_ptr<ElementDescriptor> construct(const ElementArguments &args, unsigned f) const;

  // Return characteristics for the selected set of parameters.
  std::unique_ptr<ElementCharacteristics> estimate(const ElementArguments &args) const;

private:
  std::vector<MetaElementDescriptor> library;
};

struct VerilogNodeTypePrinter final {
  VerilogNodeTypePrinter(const eda::hls::model::NodeType &type, const Library &library) :
    type(type), library(library) {}
  void print(std::ostream &out) const;

  const eda::hls::model::NodeType &type;
  const Library &library;
};

struct VerilogGraphPrinter final {
  VerilogGraphPrinter(const eda::hls::model::Graph &graph, const Library &library) :
    graph(graph), library(library) {}

  void printChan(std::ostream &out, const eda::hls::model::Chan &chan) const;
  void print(std::ostream &out) const;

  const eda::hls::model::Graph &graph;
  const Library &library;
};

std::ostream& operator <<(std::ostream &out, const VerilogNodeTypePrinter &printer);
std::ostream& operator <<(std::ostream &out, const VerilogGraphPrinter &printer);

} // namespace eda::hls::library
