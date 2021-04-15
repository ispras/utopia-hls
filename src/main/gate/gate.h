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

#include "gate/gevent.h"
#include "gate/gsymbol.h"

namespace eda {
namespace gate {

/**
 * \brief Represents a logic gate or a flip-flop/latch.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Gate final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Gate &gate);

public:
  const unsigned id() const { return _id; }
  GateSymbol gate() const { return _gate; }
  const GateEvent& clock() const { return _clock; }
  const GateEvent& reset() const { return _reset; }
  std::size_t arity() const { return _inputs.size(); }
  const Gate* input(size_t i) const { return _inputs[i]; }

  bool is_source() const { return _inputs.empty(); }
  bool is_gate() const { return !_inputs.empty() && _clock.kind() == GateEvent::ALWAYS; }
  bool is_trigger() const { return !_inputs.empty() && _clock.kind() != GateEvent::ALWAYS; }

private:
  Gate(unsigned id, GateSymbol gate, const GateEvent &clock, const GateEvent &reset,
      const std::vector<Gate *> inputs):
    _id(id), _gate(gate), _clock(clock), _reset(reset), _inputs(inputs) {}

  const unsigned _id;
  const GateSymbol _gate;
  const GateEvent _clock;
  const GateEvent _reset;
  const std::vector<Gate *> _inputs;
};

std::ostream& operator <<(std::ostream &out, const Gate &gate);

}} // namespace eda::gate

