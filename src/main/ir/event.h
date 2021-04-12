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

namespace utopia {

class VNode;

class Event final {
public:
  enum Kind {
    POSEDGE,
    NEGEDGE,
    ANYEDGE,
    LEVEL0,
    LEVEL1,
    ALWAYS,
    DELAY
  };

  static Event posedge(const VNode *signal) { return Event(POSEDGE, signal); }
  static Event negedge(const VNode *signal) { return Event(NEGEDGE, signal); }
  static Event anyedge(const VNode *signal) { return Event(ANYEDGE, signal); }
  static Event level0(const VNode *signal) { return Event(LEVEL0, signal); }
  static Event level1(const VNode *signal) { return Event(LEVEL1, signal); }
  static Event always() { return Event(ALWAYS); }
  static Event delay() { return Event(DELAY); }

private:
  Event(Kind kind, const VNode *signal = 0):
    _kind(kind), _signal(signal) {}

  const Kind _kind;
  // For edges and levels only.
  const VNode *_signal;
};

} // namespace utopia

