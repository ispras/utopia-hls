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
namespace rtl {

class VNode;

/**
 * \brief Represents a triggering event.
 * \authof <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Event final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Event &event);

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
    ALWAYS,
    /// Explicit delay: #<delay> <action>.
    DELAY
  };

  static Event posedge(const VNode *signal) {
    return Event(POSEDGE, signal);
  }

  static Event negedge(const VNode *signal) {
    return Event(NEGEDGE, signal);
  }

  static Event level0(const VNode *signal) {
    return Event(LEVEL0, signal);
  }

  static Event level1(const VNode *signal) {
    return Event(LEVEL1, signal);
  }

  static Event always() {
    return Event(ALWAYS);
  }

  static Event delay(std::size_t delay) {
    return Event(DELAY, nullptr, delay);
  }

  Kind kind() const { return _kind; }
  const VNode* signal() const { return _signal; }
  std::size_t delay() const { return _delay; }

private:
  Event(Kind kind, const VNode *signal = nullptr, std::size_t delay = 0):
    _kind(kind), _signal(signal), _delay(delay) {}

  // Event kind.
  const Kind _kind;
  // Single-bit signal for tracking events on (for edges and levels only).
  const VNode *_signal;
  // Delay value.
  const std::size_t _delay;
};

std::ostream& operator <<(std::ostream &out, const Event &event);

}} // namespace eda::rtl

