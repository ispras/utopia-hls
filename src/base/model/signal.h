//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

namespace eda::base::model {

/**
 * \brief Represents an event type.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
enum Event {
  /// Positive edge:
  ///   always_ff @(posedge <node>) begin <action> end.
  POSEDGE,
  /// Negative edge:
  ///   always_ff @(negedge <node>) begin <action> end.
  NEGEDGE,
  /// Low level:
  ///   always_latch begin if (~<node>) <action> end.
  LEVEL0,
  /// High level:
  ///   always_latch begin if (<node>) <action> end.
  LEVEL1,
  /// Continuous:
  ///   always_comb begin <action> end.
  ALWAYS,
  /// Explicit delay:
  ///   #<delay> <action>.
  DELAY
};

/**
 * \brief Represents an event-triggered signal.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
template <typename N>
class Signal final {
public:
  using List = std::vector<Signal<N>>;

  static Signal<N> posedge(N node)     { return Signal(POSEDGE, node); }
  static Signal<N> negedge(N node)     { return Signal(NEGEDGE, node); }
  static Signal<N> level0(N node)      { return Signal(LEVEL0,  node); }
  static Signal<N> level1(N node)      { return Signal(LEVEL1,  node); }
  static Signal<N> always(N node)      { return Signal(ALWAYS,  node); }
  static Signal<N> delay(size_t delay) { return Signal(delay); }

  Signal(Event event, const N node):
    _event(event), _node(node) {}

  Signal(size_t delay):
    _event(DELAY), _delay(delay) {}

  Signal():
    _event(ALWAYS) {}

  bool isEdge() const {
    return _event == POSEDGE || _event == NEGEDGE;
  }

  bool isLevel() const {
    return _event == LEVEL0 || _event == LEVEL1;
  }

  bool isAlways() const {
    return _event == ALWAYS;
  }

  bool isDelay() const {
    return _event == DELAY;
  }

  Event event()  const { return _event; }
  const N node() const { return _node;  }
  size_t delay() const { return _delay; }

  bool operator ==(const Signal<N> &rhs) const {
    if (&rhs == this) {
      return true;
    }

    return _event == rhs._event
        && _node  == rhs._node
        && _delay == rhs._delay;
  }

private:
  // Event kind.
  Event _event;
  // Node for tracking events on.
  N _node;
  // Delay value.
  size_t _delay;
};

inline std::ostream &operator <<(std::ostream &out, Event event) {
  switch (event) {
  case POSEDGE:
    return out << "posedge";
  case NEGEDGE:
    return out << "negedge";
  case LEVEL0:
    return out << "level0";
  case LEVEL1:
    return out << "level1";
  case ALWAYS:
    return out << "*";
  case DELAY:
    return out << "#";
  }
  return out;
}

template <typename N>
inline std::ostream &operator <<(std::ostream &out, const Signal<N> &signal) {
  if (signal.event() == ALWAYS) {
    return out << "*";
  }
  if (signal.event() == DELAY) {
    return out << "#" << signal.delay();
  }
  return out << signal.event() << "(" << signal.node() << ")";
}

} // namespace eda::base::model
