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

#include <iostream>
#include <vector>

#include "gate/gsymbol.h"
#include "gate/signal.h"

namespace eda {
namespace gate {

class Netlist;

/**
 * \brief Represents a logic gate or a flip-flop/latch.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Gate final {
  // Creation.
  friend class Netlist;
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Gate &gate);

public:
  const unsigned id() const { return _id; }
  GateSymbol gate() const { return _gate; }
  std::size_t arity() const { return _inputs.size(); }
  const Signal& input(size_t i) const { return _inputs[i]; }

  bool is_source() const { return _inputs.empty(); }
  bool is_trigger() const { return is_sequential(); }
  bool is_gate() const { return !is_source() && !is_trigger(); }

private:
  Gate(unsigned id):
    _id(id), _gate(GateSymbol::NOP), _inputs({}) {}

  Gate(unsigned id, GateSymbol gate, const std::vector<Signal> inputs):
    _id(id), _gate(gate), _inputs(inputs) {}

  bool is_sequential() const {
    for (const auto &input: _inputs) {
      if (input.kind() != Signal::ALWAYS) {
        return true;
      }
    }
    return false;
  }

  const unsigned _id;
  const GateSymbol _gate;
  const std::vector<Signal> _inputs;
};

std::ostream& operator <<(std::ostream &out, const Gate &gate);

}} // namespace eda::gate

