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
#include <map>
#include <memory>
#include <string>

using namespace eda::util;
using namespace eda::hls::model;

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
  Parameter(const std::string &name,
            const Constraint &constraint,
            unsigned value):
    name(name), constraint(constraint), value(value) {}

  Parameter(const Parameter &) = default;

  const std::string name;
  const Constraint constraint;
  unsigned value;
};

struct Parameters final {
  Parameters() = default;
  Parameters(const Parameters &) = default;

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

  std::map<std::string, Parameter> params;
};

/// RTL port with name, direction, and width.
struct Port {
  enum Direction { IN, OUT, INOUT };

  Port(const std::string &name, const Direction &direction,
       unsigned latency, unsigned width, bool isParam, char param):
    name(name),
    direction(direction),
    latency(latency),
    width(width),
    isParam(isParam),
    param(param) {}
  Port(const Port &port):
    name(port.name),
    direction(port.direction),
    latency(port.latency),
    width(port.width),
    isParam(port.isParam),
    param(port.param) {}

  const std::string name;
  const Direction direction;
  const unsigned latency;
  const unsigned width;
  const bool isParam;
  const char param;
};

/// Description of a constructed element (module).
struct Element final {
  // TODO: Code, Path, etc.
  explicit Element(const std::vector<Port> &ports): ports(ports) {}

  // TODO add mutual relation between spec ports and impl ports
  const std::vector<Port> ports;

  // TODO there should be different IRs: MLIR FIRRTL or Verilog|VHDL described in FIRRTL
  std::string ir;

  // TODO path
  std::string path;
};

/// Description of a parameterized constructor of elements.
struct MetaElement {
  MetaElement(const std::string &name,
              const Parameters &params,
              const std::vector<Port> &ports):
      name(name), params(params), ports(ports) {}

  /// Estimates the indicators the given set of parameters.
  virtual void estimate(const Parameters &params,
                        Indicators &indicators) const = 0;

  virtual std::unique_ptr<Element> construct(const Parameters &params) const = 0;

  const std::string name;
  const Parameters params;
  const std::vector<Port> ports;
};

class Library final : public Singleton<Library> {
  friend class Singleton<Library>;

public:
  void initialize(const std::string &libPath, const std::string &catalogPath);
  void finalize();

  /// Searches for a meta-element for the given node type.
  std::shared_ptr<MetaElement> find(const NodeType &nodetype);
  std::shared_ptr<MetaElement> create(const NodeType &nodetype);
  /// Searches for a meta-element for the given name.
  //std::shared_ptr<MetaElement> find(const std::string &name);

  void add(const std::shared_ptr<MetaElement> &metaElement) {
    cache.push_back(metaElement);
  }

private:
  Library() {}

  /// Cached meta-elements.
  std::vector<std::shared_ptr<MetaElement>> cache;
};

} // namespace eda::hls::library
