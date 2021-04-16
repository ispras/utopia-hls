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
 * \brief Defines names of supported logical gates and flip-flops/latches.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
enum GateSymbol {
  //----------------------------------------------------------------------------
  // Logic gates
  //----------------------------------------------------------------------------

  /// Identity: OUT = X.
  NOP,
  /// Negation: OUT = ~X.
  NOT,
  /// Conjunction: OUT = X & Y (& ...).
  AND,
  /// Disjunction: OUT = X | Y (| ...).
  OR,
  /// Exclusive OR: OUT = X + Y (+ ...) (mod 2).
  XOR,
  /// Sheffer's stroke: OUT = ~(X & Y (& ...)).
  NAND,
  /// Peirce's arrow: OUT <= ~(X | Y (| ...)).
  NOR,
  /// Exclusive NOR: OUT <= ~(X + Y (+ ...) (mod 2)).
  XNOR,

  //----------------------------------------------------------------------------
  // Flip-flops and latches
  //----------------------------------------------------------------------------

  /// D latch (Q, D, ENA):
  /// Q(t) = ENA(level1) ? Q(t-1) : D.
  LATCH,
  /// D flip-flop (Q, D, CLK):
  /// Q(t) = CLK(posedge) ? D : Q(t-1).
  DFF,
  /// D flip-flop w/ (asynchronous) reset and set (Q, D, CLK, RST, SET):
  /// Q(t) = RST(level1) ? 0 : (SET(level1) ? 1 : (CLK(posedge) ? D : Q(t-1))).
  DFFrs
};

std::ostream& operator <<(std::ostream &out, GateSymbol gate);

}} // namespace eda::gate

