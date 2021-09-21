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

#include "gate/model/gsymbol.h"
#include "gate/model/signal.h"

namespace eda::rtl::compiler {
  class Compiler;
} // eda::rtl::compiler

namespace eda::gate::model {

class Netlist;

/**
 * \brief Represents a logic gate or a flip-flop/latch.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Gate final {
  // Creation.
  friend class Netlist;
  friend class eda::rtl::compiler::Compiler;

public:
  using List = std::vector<Gate *>;

  const unsigned id() const { return _id; }
  GateSymbol kind() const { return _kind; }
  std::size_t arity() const { return _inputs.size(); }

  const Signal::List& inputs() const { return _inputs; }
  const Signal& input(size_t i) const { return _inputs[i]; }

  bool is_source() const { return _kind == GateSymbol::NOP && _inputs.empty(); }
  bool is_value() const { return _kind == GateSymbol::ONE || _kind == GateSymbol::ZERO; }
  bool is_trigger() const { return is_sequential(); }
  bool is_gate() const { return !is_source() && !is_trigger(); }

private:
  Gate(unsigned id):
    _id(id), _kind(GateSymbol::NOP), _inputs() {}

  Gate(unsigned id, GateSymbol gate, const Signal::List inputs):
    _id(id), _kind(gate), _inputs(inputs) {}

  bool is_sequential() const {
    for (const auto &input: _inputs) {
      if (input.kind() != Event::ALWAYS) {
        return true;
      }
    }
    return false;
  }

  void set_kind(GateSymbol kind) { _kind = kind; }

  void set_inputs(const Signal::List &inputs) {
    _inputs.assign(inputs.begin(), inputs.end());
  }

  const unsigned _id;

  GateSymbol _kind;
  Signal::List _inputs;
};

std::ostream& operator <<(std::ostream &out, const Gate &gate);

} // namespace eda::gate::model
