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

namespace eda {
namespace rtl {

class VNode;

/**
 * \brief Represents a triggering event.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Event final {
public:
  typedef std::vector<Event> List;

  enum Kind {
    /// Positive edge: always_ff @(posedge <node>) begin <action> end.
    POSEDGE,
    /// Negative edge: always_ff @(negedge <node>) begin <action> end.
    NEGEDGE,
    /// Low level: always_latch begin if (~<node>) <action> end.
    LEVEL0,
    /// High Level: always_latch begin if (<node>) <action> end.
    LEVEL1,
    /// Continuous: always_comb begin <action> end.
    ALWAYS,
    /// Explicit delay: #<delay> <action>.
    DELAY
  };

  static Event posedge(const VNode *node) {
    return Event(POSEDGE, node);
  }

  static Event negedge(const VNode *node) {
    return Event(NEGEDGE, node);
  }

  static Event level0(const VNode *node) {
    return Event(LEVEL0, node);
  }

  static Event level1(const VNode *node) {
    return Event(LEVEL1, node);
  }

  static Event always() {
    return Event(ALWAYS);
  }

  static Event delay(std::size_t delay) {
    return Event(DELAY, nullptr, delay);
  }

  bool edge() const { return _kind == POSEDGE || _kind == NEGEDGE; }
  bool level() const { return _kind == LEVEL0 || _kind == LEVEL1; }

  Kind kind() const { return _kind; }
  const VNode* node() const { return _node; }
  std::size_t delay() const { return _delay; }

  bool operator ==(const Event &rhs) const {
    if (&rhs == this) {
      return true;
    }

    return _kind == rhs._kind && _node == rhs._node && _delay == rhs._delay;
  }

private:
  Event(Kind kind, const VNode *node = nullptr, std::size_t delay = 0):
    _kind(kind), _node(node), _delay(delay) {}

  // Event kind.
  const Kind _kind;
  // Single-bit node for tracking events on (for edges and levels only).
  const VNode *_node;
  // Delay value.
  const std::size_t _delay;
};

std::ostream& operator <<(std::ostream &out, const Event::Kind &kind);
std::ostream& operator <<(std::ostream &out, const Event &event);

}} // namespace eda::rtl

