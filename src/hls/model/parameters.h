//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/constraint.h"

#include <cassert>
#include <map>
#include <string>

namespace eda::hls::model {

class Parameter final {
public:
  Parameter(const std::string &name,
            const Constraint<unsigned> &constraint,
            unsigned value):
    name(name), constraint(constraint), value(value), known(true) {}

  Parameter(const std::string &name,
            const Constraint<unsigned> &constraint):
    name(name), constraint(constraint), value(-1u), known(false) {}

  Parameter(const std::string &name, unsigned value):
    name(name), constraint(), value(value), known(true) {}

  Parameter(const std::string &name):
    name(name), constraint(), value(-1u), known(false) {}

  Parameter(const Parameter &) = default;

  std::string getName() const {
    return name;
  }

  bool isKnown() const {
    return known;
  }

  unsigned getValue() const {
    assert(known);
    return value;
  }

  unsigned getMax() const {
    return constraint.getMax();
  }

  unsigned getMin() const {
    return constraint.getMin();
  }

  void setValue(unsigned newValue) {
    //assert(constraint.check(newValue));
    value = newValue;
    known = true;
  }

  void resetValue() {
    value = -1u;
    known = false;
  }

private:
  const std::string name;
  const Constraint<unsigned> constraint;
  unsigned value;
  bool known;
};

class Parameters final {
public:
  Parameters() = default;
  Parameters(const Parameters &) = default;

  const std::map<std::string, Parameter>& getAll() const {
    return params;
  }

  bool contains(const std::string &name) const {
    return params.find(name) != params.end();
  }

  Parameter get(const std::string &name) const {
    const auto &param = find(name);
    return param;
  }

  void add(const Parameter &param) {
    params.insert({ param.getName(), param });
  }

  unsigned getValue(const std::string &name) const {
    const auto &param = find(name);
    return param.getValue();
  }

  void setValue(const std::string &name, unsigned value) {
    auto &param = find(name);
    param.setValue(value);
  }

  void resetValue(const std::string &name) {
    auto &param = find(name);
    param.resetValue();
  }

private:
  const Parameter& find(const std::string &name) const {
    auto i = params.find(name);
    assert(i != params.end() && "Parameter is not found");
    return i->second;
  }

  Parameter& find(const std::string &name) {
    auto i = params.find(name);
    assert(i != params.end() && "Parameter is not found");
    return i->second;
  }

  std::map<std::string, Parameter> params;
};

} // namespace eda::hls::model
