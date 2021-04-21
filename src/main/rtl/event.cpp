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

#include "rtl/event.h"
#include "rtl/vnode.h"

namespace eda::rtl {

std::ostream& operator <<(std::ostream &out, const Event::Kind &kind) {
  switch (kind) {
  case Event::POSEDGE:
    return out << "posedge";
  case Event::NEGEDGE:
    return out << "negedge";
  case Event::LEVEL0:
    return out << "level0";
  case Event::LEVEL1:
    return out << "level1";
  case Event::ALWAYS:
    return out << "*";
  case Event::DELAY:
    return out << "#";
  }
  return out;
}

std::ostream& operator <<(std::ostream &out, const Event &event) {
  if (event.kind() == Event::ALWAYS) {
    return out << "*";
  }
  if (event.kind() == Event::DELAY) {
    return out << "#" << event.delay();
  }
  return out << event.kind() << "(" << event.node()->name() << ")";
}

} // namespace eda::rtl
