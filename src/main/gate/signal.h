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
#include <vector>

#include "rtl/event.h"

using namespace eda::rtl;

namespace eda::gate {

class Gate;

/**
 * \brief Represents a triggering signal.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Signal final {
public:
  typedef std::vector<Signal> List;

  Signal(Event::Kind kind, const Gate *gate):
      _kind(kind), _gate(gate) {
    assert(gate != nullptr);
  }

  bool edge() const { return _kind == Event::POSEDGE || _kind == Event::NEGEDGE; }
  bool level() const { return _kind == Event::LEVEL0 || _kind == Event::LEVEL1; }

  Event::Kind kind() const { return _kind; }
  const Gate* gate() const { return _gate; }

private:
  Event::Kind _kind;
  const Gate *_gate;
};

std::ostream& operator <<(std::ostream &out, const Signal &signal);

} // namespace eda::gate
