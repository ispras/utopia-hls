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

namespace utopia {

/**
 * \brief Defines names of supported BV operations.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
enum Function {
  /// Identity function: y = x.
  NOP,
  /// Multiplexer (selector): y = mux(c[0] -> x[0], ..., c[n-1] -> x[n-1]).
  MUX,
  /// Addition: y = x[0] + x[1].
  ADD,
  /// Subtraction: y = x[0] - x[1].
  SUB,
  /// Multiplication: y = x[0] * x[1].
  MUL,
  /// Division: y = x[0] / x[1].
  DIV
  // TODO: Add more operations.
};

} // namespace utopia

