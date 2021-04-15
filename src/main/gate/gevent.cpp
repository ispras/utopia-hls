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
#include "gate/gevent.h"

namespace eda {
namespace gate {

std::ostream& operator <<(std::ostream &out, const GateEvent &event) {
  switch (event.kind()) {
  case GateEvent::POSEDGE:
    return out << "posedge(" << event.signal()->id() << ")";
  case GateEvent::NEGEDGE:
    return out << "negedge(" << event.signal()->id() << ")";
  case GateEvent::LEVEL0:
    return out << "level0(" << event.signal()->id() << ")";
  case GateEvent::LEVEL1:
    return out << "level1(" << event.signal()->id() << ")";
  case GateEvent::ALWAYS:
    return out << "*";
  }

  return out;
}

}} // namespace eda::gate

