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

#include "gate/model/gsymbol.h"

namespace eda::gate::model {

std::ostream& operator <<(std::ostream &out, GateSymbol gate) {
  switch (gate) {
  case GateSymbol::ZERO:
    return out << "0";
  case GateSymbol::ONE:
    return out << "1";
  case GateSymbol::NOP:
    return out << "buf";
  case GateSymbol::NOT:
    return out << "not";
  case GateSymbol::AND:
    return out << "and";
  case GateSymbol::OR:
    return out << "or";
  case GateSymbol::XOR:
    return out << "xor";
  case GateSymbol::NAND:
    return out << "nand";
  case GateSymbol::NOR:
    return out << "nor";
  case GateSymbol::XNOR:
    return out << "xnor";
  case GateSymbol::LATCH:
    return out << "latch";
  case GateSymbol::DFF:
    return out << "dff";
  case GateSymbol::DFFrs:
    return out << "dff_rs";
  }

  return out;
}

} // namespace eda::gate::model
