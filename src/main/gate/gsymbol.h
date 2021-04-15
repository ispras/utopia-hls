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

namespace eda {
namespace gate {

/**
 * \brief Defines names of supported logical gates.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
enum GateSymbol {
  /// Identity: y <= x.
  NOP,
  /// Negation: y <= ~x.
  NOT,
  /// Conjunction: y <= x[0] & x[1].
  AND,
  /// Disjunction: y <= x[0] | x[1].
  OR,
  /// Exclusive OR: y <= x[0] + x[1] (mod 2).
  XOR,
  /// Sheffer's stroke: y <= ~(x[0] & x[1]).
  NAND,
  /// Peirce's arrow: y <= ~(x[0] | x[1]).
  NOR,
  /// Exclusive NOR: y <= ~(x[0] + x[1] (mod 2)).
  XNOR
};

std::ostream& operator <<(std::ostream &out, GateSymbol gate);

}} // namespace eda::gate

