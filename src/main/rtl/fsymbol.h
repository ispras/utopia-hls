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
namespace rtl {

/**
 * \brief Defines names of supported RTL-level functions.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
enum FuncSymbol {
  /// Identity: OUT = X.
  NOP,
  /// Negation: OUT = ~X.
  NOT,
  /// Disjunction: OUT = X | Y.
  OR,
  /// Conjunction: OUT = X & Y.
  AND, 
  /// Addition: OUT = X + Y.
  ADD,
  /// Subtraction: OUT = X - Y.
  SUB,
  /// Multiplication: OUT = X * Y.
  MUL,
  /// Division: OUT = X / Y.
  DIV,
  // TODO: Add more functions.
  /// Multiplexor: OUT = MUX(C[1], ..., C[n]; X[1], ..., X[n]).
  MUX
};

std::ostream& operator <<(std::ostream &out, FuncSymbol func);

}} // namespace eda::rtl

