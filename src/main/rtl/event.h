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

namespace eda::rtl {

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

  Event(Kind kind, const VNode *node = nullptr):
    _kind(kind), _node(node), _delay(0) {}

  Event(std::size_t delay):
    _kind(DELAY), _node(nullptr), _delay(delay) {}

  Event():
    _kind(ALWAYS), _node(nullptr), _delay(0) {}

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
  // Event kind.
  const Kind _kind;
  // Single-bit node for tracking events on (for edges and levels only).
  const VNode *_node;
  // Delay value.
  const std::size_t _delay;
};

std::ostream& operator <<(std::ostream &out, const Event::Kind &kind);
std::ostream& operator <<(std::ostream &out, const Event &event);

} // namespace eda::rtl
