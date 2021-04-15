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

namespace eda {
namespace gate {

class Gate;

/**
 * \brief Represents a triggering event.
 * \authof <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class GateEvent final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const GateEvent &event);

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

  static GateEvent posedge(const Gate *signal) {
    return GateEvent(POSEDGE, signal);
  }

  static GateEvent negedge(const Gate *signal) {
    return GateEvent(NEGEDGE, signal);
  }

  static GateEvent level0(const Gate *signal) {
    return GateEvent(LEVEL0, signal);
  }

  static GateEvent level1(const Gate *signal) {
    return GateEvent(LEVEL1, signal);
  }

  static GateEvent always() {
    return GateEvent(ALWAYS);
  }

  Kind kind() const { return _kind; }
  const Gate* signal() const { return _signal; }

private:
  GateEvent(Kind kind, const Gate *signal = nullptr):
    _kind(kind), _signal(signal) {}

  // Event kind.
  const Kind _kind;
  // Single-bit signal for tracking events on (for edges and levels only).
  const Gate *_signal;
};

std::ostream& operator <<(std::ostream &out, const GateEvent &event);

}} // namespace eda::gate

