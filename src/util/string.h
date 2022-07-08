//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace eda::utils {

inline bool starts_with(const std::string &string, const std::string &prefix) {
  return string.size() >= prefix.size()
      && string.compare(0, prefix.size(), prefix) == 0;
}

inline bool ends_with(const std::string &string, const std::string &suffix) {
  return string.size() >= suffix.size()
      && string.compare(string.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

inline std::string replaceSomeChars(const std::string &buf) {
  std::string result = buf;
  std::replace(result.begin(), result.end(), '$', '_');
  std::replace(result.begin(), result.end(), ',', '_');
  std::replace(result.begin(), result.end(), '>', '_');
  std::replace(result.begin(), result.end(), '<', '_');
  return result;
}

template<typename... Args>
std::string format(const std::string &format, Args... args) {
  int length = snprintf(nullptr, 0, format.c_str(), args ...);

  if (length < 0) {
    throw std::runtime_error("Formatting error");
  }

  auto size = static_cast<size_t>(length + 1);
  auto buffer = std::make_unique<char[]>(size);

  snprintf(buffer.get(), size, format.c_str(), args ...);
  return std::string(buffer.get(), buffer.get() + size - 1);
}

std::string unique_name(const std::string &prefix);

} // namespace eda::utils
