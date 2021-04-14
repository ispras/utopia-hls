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

#include <memory>
#include <stdexcept>
#include <string>

namespace eda {
namespace ir {
namespace utils {

template<typename ... Args>
std::string format(const std::string &format, Args ... args) {
  int length = snprintf(nullptr, 0, format.c_str(), args ...);

  if (length < 0) {
    throw std::runtime_error("Formatting error");
  }

  auto size = static_cast<size_t>(length + 1);
  auto buffer = std::make_unique<char[]>(size);

  snprintf(buffer.get(), size, format.c_str(), args ...);
  return std::string(buffer.get(), buffer.get() + size - 1);
}

}}} // namespace eda::ir::utils

