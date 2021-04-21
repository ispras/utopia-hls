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

#include <iostream>

#include "gate/gate.h"

namespace eda::gate {

static std::ostream& operator <<(std::ostream &out, const Signal::List &signals) {
  bool separator = false;
  for (const Signal &signal: signals) {
    out << (separator ? ", " : "") << signal.kind() << "(" << signal.gate()->id() << ")";
    separator = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const Gate &gate) {
  if (gate.is_source()) {
    return out << "S{" << gate.id() << "}";
  } else {
    return out << (gate.is_gate() ? "G" : "T") << "{"
               << gate.id() << " <= " << gate.kind() << "(" << gate.inputs() << ")}";
  }
}

} // namespace eda::gate
