//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"
#include "util/singleton.h"

#include <cassert>
#include <iostream>
#include <map>

using namespace eda::util;

namespace eda::hls::library {

enum Indicator {
  Frequency,
  Throughput,
  Latency,
  Power,
  Area
};

struct Indicators final {
  unsigned frequency;
  unsigned throughput;
  unsigned latency;
  unsigned power;
  unsigned area;
};

struct Constraint final {
  Constraint(unsigned min, unsigned max):
    min(min), max(max) {}

  Constraint(const Constraint &) = default;

  bool check(unsigned value) const {
    return min <= value && value <= max;
  }

  const unsigned min;
  const unsigned max;
};

struct Parameter final {
  Parameter(const std::string &name, const Constraint &constraint,
            unsigned value):
    name(name), constraint(constraint), value(value) {}

  Parameter(const Parameter &) = default;

  const std::string name;
  const Constraint constraint;
  unsigned value;
};

struct Parameters final {
  explicit Parameters(const std::string &elementName):
    elementName(elementName) {}

  Parameters(const Parameters &) = default;

  Parameters(const std::string &elementName, const Parameters &params):
    elementName(elementName), params(params.params) {}

  Parameter get(const std::string &name) const {
    auto i = params.find(name);
    assert(i != params.end() && "Parameter is not found");
    return i->second;
  }

  void add(const Parameter &param) {
    params.insert({ param.name, param });
  }

  unsigned value(const std::string &name) const {
    auto i = params.find(name);
    assert(i != params.end() && "Parameter is not found");
    return i->second.value;
  }

  void set(const std::string &name, unsigned value) {
    auto i = params.find(name);
    assert(i != params.end() && "Parameter is not found");
    i->second.value = value;
  }

  const std::string elementName;
  std::map<std::string, Parameter> params;
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

/// Description of a class of modules with the given names,
///  parameter list, and allowed ranges of their values.
struct MetaElement final {
  MetaElement(const std::string &name, const Parameters &params,
              const Ports &ports):
    name(name), params(params), ports(ports) {}
  MetaElement(const MetaElement &) = default;

  const std::string name;
  const Parameters params;
  const Ports ports;
};

/// Description of a module with the given name and values of parameters.
struct Element final {
  explicit Element(const Ports &ports): ports(ports) {}

  // TODO add mutual relation between spec ports and impl ports
  const Ports ports;

  // TODO there should be different IRs: MLIR FIRRTL or Verilog|VHDL described in FIRRTL
  std::string ir;
};

class Library final : public Singleton<Library> {
public:
  Library();

  /// Return a list of parameters for the module with the given name
  /// (and correspodent functionality).
  const MetaElement& find(const eda::hls::model::NodeType &type);

  /// Return a list of parameters iff the element is in the library.
  const MetaElement& find(const std::string &name) const;

  /// Return a module with the selected set of parameters
  /// (where f is an additional parameter).
  std::unique_ptr<Element> construct(const MetaElement &meta) const;

  /// Return characteristics for the selected set of parameters.
  void estimate(const Parameters &params, Indicators &indicators) const;

private:
  /// NodeType => MetaElement if the element is missing in the library.
  MetaElement createMetaElement(const eda::hls::model::NodeType &type) const;
  /// MetaElement to populate the library.
  MetaElement createMetaElement(const std::string &name) const;
  std::vector<MetaElement> library;
};

class VerilogNodeTypePrinter final {
public:
  VerilogNodeTypePrinter(const eda::hls::model::NodeType &type):
    type(type) {}
  void print(std::ostream &out) const;

private:
  const eda::hls::model::NodeType &type;
};

class VerilogGraphPrinter final {
public:
  VerilogGraphPrinter(const eda::hls::model::Graph &graph):
    graph(graph) {}

  void print(std::ostream &out) const;

private:
  void printChan(std::ostream &out, const eda::hls::model::Chan &chan) const;
  const eda::hls::model::Graph &graph;
};

std::ostream& operator <<(std::ostream &out, const VerilogNodeTypePrinter &printer);
std::ostream& operator <<(std::ostream &out, const VerilogGraphPrinter &printer);

} // namespace eda::hls::library
