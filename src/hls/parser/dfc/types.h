//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "util/string.h"

#include <cstddef>
#include <sstream>
#include <string>

namespace dfc {

class type {
public:
  virtual std::string name() const = 0;
  virtual std::size_t size() const = 0;
};

class scalar: public type {};

class untyped: public scalar {};

template<std::size_t IntBits, std::size_t FracBits, bool IsSigned>
class fix: public scalar {
public:
  static constexpr std::size_t type_size = IntBits + FracBits;

  static constexpr std::string type_name() {
    return "fix<" + std::to_string(IntBits)  + ","
                  + std::to_string(FracBits) + ","
                  + std::to_string(IsSigned) + ">";
  }
 
  constexpr std::string name() const override { return type_name(); }
  constexpr std::size_t size() const override { return type_size; }
};

template<std::size_t IntBits, std::size_t FracBits>
class sfix: public fix<IntBits, FracBits, true> {};

template<std::size_t Bits>
class sint: public sfix<Bits, 0> {};

using sint8  = sint<8>;
using sint16 = sint<16>;
using sint32 = sint<32>;
using sint64 = sint<64>;

template<std::size_t IntBits, std::size_t FracBits>
class ufix: public fix<IntBits, FracBits, false> {};

template<std::size_t Bits>
class uint: public ufix<Bits, 0> {};

using bit    = uint<1>;
using uint8  = uint<8>;
using uint16 = uint<16>;
using uint32 = uint<32>;
using uint64 = uint<64>;

template<std::size_t ExpBits, std::size_t Precision>
class real: public scalar {
public:
  static constexpr std::size_t type_size = ExpBits + Precision;

  static constexpr std::string type_name() {
    return "real<" + std::to_string(ExpBits) + ","
                   + std::to_string(Precision) + ">";
  }

  constexpr std::string name() const override { return type_name(); }
  constexpr std::size_t size() const override { return type_size; }
};

using float16 = real<5, 11>;
using float32 = real<8, 24>;
using float64 = real<11, 53>;

template<std::size_t Bits>
class bits: public scalar {
  static constexpr std::size_t type_size = Bits;

  static constexpr std::string type_name() {
    return "bits<" + std::to_string(Bits) + ">";
  }

  constexpr std::size_t size() const override { return type_size; }
};

class composite: public type {};

template<typename Head, typename... Tail>
class tuple: public composite {
public:
  static constexpr std::size_t type_size = Head::type_size + (... + Tail::type_size);

  static constexpr std::string type_name() {
    std::stringstream out;
    out << "tuple<" << Head::type_name();
    ((out << "," << Tail::type_name()), ...);
    out << ">";
    return out.str();
  }

  constexpr std::string name() const override { return type_name(); }
  constexpr std::size_t size() const override { return type_size; }
};

template<typename Type>
class complex: public tuple<Type, Type> {
public:
  static constexpr std::string type_name() {
    return "complex<" + Type::type_name() + ">";
  }

  constexpr std::string name() const override { return type_name(); }
};

template<typename Type, std::size_t... Sizes>
class tensor: public composite {
public:
  static constexpr std::size_t type_size = Type::type_size * (... * Sizes);

  static constexpr std::string type_name() {
    std::stringstream out;
    out << "tensor<" << Type::type_name();
    ((out << "," << Sizes), ...);
    out << ">";
    return out.str();
  }

  constexpr std::string name() const override { return type_name(); }
  constexpr std::size_t size() const override { return type_size; }
};

template<typename Type, std::size_t Size>
class vector: public tensor<Type, Size> {};

template<typename Type, std::size_t Rows, std::size_t Cols>
class matrix: public tensor<Type, Rows, Cols> {};

} // namespace dfc

