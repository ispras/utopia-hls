/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

#pragma once

#include <cassert>
#include <iostream>
#include <vector>

#include "gate/gate.h"

namespace eda {
namespace gate {

class Net;

/**
 * \brief Represents a gate-level netlist.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Netlist final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Netlist &netlist);

public:
  typedef typename std::vector<Gate *>::const_iterator const_iterator;

  Netlist() {
    _gates.reserve(1024*1024);
  } 

  std::size_t size() const { return _gates.size(); }
  const_iterator begin() const { return _gates.cbegin(); }
  const_iterator end() const { return _gates.cend(); }

  /// Create an input.
  Gate* add_source(unsigned id) {
    return add_gate(new Gate(id, GateSymbol::NOP, Event::always(), Event::always(), {}));
  }

  /// Creates a logical gate (combinational).
  Gate* add_gate(unsigned id, GateSymbol gate, const std::vector<Gate *> inputs) {
    return add_gate(new Gate(id, gate, GateEvent::always(), GateEvent::always(), inputs));
  }

  /// Creates and adds an f-node (s = function).
  Gate* add_trigger(unsigned id, GateSymbol gate, const GateEvent &clock,
      const GateEvent &reset, Gate *data) {
    return add_gate(new Gate(id, gate, clock, reset, { data }));
  }

  /// Synthesizes the gate-level netlist from the RTL-level net.
  void create(const Net &net);

private:
  Gate* add_gate(Gate *gate) {
    _gates.push_back(gate);
    return gate;
  }

  std::vector<Gate *> _gates;
};

std::ostream& operator <<(std::ostream &out, const Netlist &netlist);

}} // namespace eda::gate

