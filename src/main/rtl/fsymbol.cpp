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

#include "rtl/fsymbol.h"

namespace eda::rtl {

std::ostream& operator <<(std::ostream &out, FuncSymbol func) {
  switch (func) {
  case FuncSymbol::NOP:
    return out << "";
  case FuncSymbol::NOT:
    return out << "~";
  case FuncSymbol::AND:
    return out << "&";
  case FuncSymbol::OR:
    return out << "|";
  case FuncSymbol::XOR:
    return out << "^";
  case FuncSymbol::ADD:
    return out << "+";
  case FuncSymbol::SUB:
    return out << "-";
  case FuncSymbol::MUX:
    return out << "mux";
  }
  return out;
}

} // namespace eda::rtl
