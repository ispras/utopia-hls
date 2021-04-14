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

#include "function.h"

namespace eda {
namespace ir {

std::ostream& operator <<(std::ostream &out, Function fun) {
  switch (fun) {
  case Function::NOP:
    return out << "";
  case Function::NOT:
    return out << "~";
  case Function::ADD:
    return out << "+";
  case Function::SUB:
    return out << "-";
  case Function::MUL:
    return out << "*";
  case Function::DIV:
    return out << "/";
  }

  return out;
}

}} // namespace eda::ir

