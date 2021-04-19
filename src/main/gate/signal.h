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

namespace eda {
namespace gate {

class Gate;

/**
 * \brief Represents a triggering signal.
 * \authof <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Signal final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Signal &signal);

public:
  enum Kind {
    /// Positive edge: always_ff @(posedge <signal>) begin <action> end.
    POSEDGE,
    /// Negative edge: always_ff @(negedge <signal>) begin <action> end.
    NEGEDGE,
    /// Low level: always_latch begin if (~<signal>) <action> end.
    LEVEL0,
    /// High Level: always_latch begin if (<signal>) <action> end.
    LEVEL1,
    /// Continuous: always_comb begin <action> end.
    ALWAYS
  };

  static Signal posedge(const Gate *gate) {
    return Signal(POSEDGE, gate);
  }

  static Signal negedge(const Gate *gate) {
    return Signal(NEGEDGE, gate);
  }

  static Signal level0(const Gate *gate) {
    return Signal(LEVEL0, gate);
  }

  static Signal level1(const Gate *gate) {
    return Signal(LEVEL1, gate);
  }

  static Signal always(const Gate *gate) {
    return Signal(ALWAYS, gate);
  }

  Kind kind() const { return _kind; }
  const Gate* gate() const { return _gate; }

private:
  Signal(Kind kind, const Gate *gate):
      _kind(kind), _gate(gate) {
    assert(gate != nullptr);
  }

  Kind _kind;
  const Gate *_gate;
};

std::ostream& operator <<(std::ostream &out, const Signal::Kind &kind);
std::ostream& operator <<(std::ostream &out, const Signal &signal);

}} // namespace eda::gate

