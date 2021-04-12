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

namespace utopia {

class Type final {
public:
  enum Kind {
    SINT,
    UINT,
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

class Variable final {
public:
  enum Kind {
    WIRE,
    REG
  };

  enum Dir {
    INPUT,
    OUTPUT,
    INNER
  };

  static Variable var(const std::string &name, Kind kind, Dir dir, const Type &type) {
    return Variable(name, kind, dir, type);
  }

  const std::string& name() const { return _name; }
  Kind kind() const { return _kind; }
  Dir dir() const { return _dir; }
  const Type& type() const { return _type; }

private:
  Variable(const std::string &name, Kind kind, Dir dir, const Type &type):
    _name(name), _kind(kind), _dir(dir), _type(type) {}

  const std::string _name;
  const Kind _kind;
  const Dir _dir;
  const Type _type;
};

} // namespace utopia

