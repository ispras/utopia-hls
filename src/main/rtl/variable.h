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

#include <cstddef>
#include <iostream>

namespace eda {
namespace rtl {

/**
 * \brief Represents a data type.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Type final {
public:
  enum Kind {
    /// Signed integer (width).
    SINT,
    /// Unsigned integer (width).
    UINT,
    /// Floating-point number (width, fract).
    REAL
  };

  static Type sint(std::size_t width) { return Type(SINT, width); }
  static Type uint(std::size_t width) { return Type(UINT, width); }
  static Type real(std::size_t width, std::size_t fract) { return Type(REAL, width, fract); }

  Kind kind() const { return _kind; }
  std::size_t width() const { return _width; }
  std::size_t fract() const { return _fract; }

private:
  Type(Kind kind, std::size_t width, std::size_t fract = 0):
    _kind(kind), _width(width), _fract(fract) {}

  const Kind _kind;
  const std::size_t _width;
  const std::size_t _fract;
};

/**
 * \brief Represents a variable (wire or register).
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Variable final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Variable &variable);

public:
  enum Kind {
    WIRE,
    REG
  };

  enum Bind {
    INPUT,
    OUTPUT,
    INNER
  };

  Variable(const std::string &name, Kind kind, Bind bind, const Type &type):
    _name(name), _kind(kind), _bind(bind), _type(type) {}

  Variable(const std::string &name, Kind kind, const Type &type):
    _name(name), _kind(kind), _bind(INNER), _type(type) {}

  const std::string& name() const { return _name; }
  Kind kind() const { return _kind; }
  Bind bind() const { return _bind; }
  const Type& type() const { return _type; }

private:
  const std::string _name;
  const Kind _kind;
  const Bind _bind;
  const Type _type;
};

inline std::ostream& operator <<(std::ostream &out, const Variable &variable) {
  return out << variable.name();
}

}} // namespace eda::rtl

