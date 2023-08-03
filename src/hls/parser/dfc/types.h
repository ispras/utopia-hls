//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"
#include "utils/singleton.h"
#include "utils/string.h"

#include <cstddef>
#include <sstream>
#include <string>

namespace dfc {

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

struct type {
  virtual std::string name()            const = 0;
  virtual std::size_t size()            const = 0;
  virtual eda::hls::model::Type &info() const = 0;
};

//===----------------------------------------------------------------------===//
// Basic
//===----------------------------------------------------------------------===//

struct basic: public type {};

//===----------------------------------------------------------------------===//
// Untyped
//===----------------------------------------------------------------------===//

struct untyped: public basic {};

//===----------------------------------------------------------------------===//
// Fixed
//===----------------------------------------------------------------------===//

template<std::size_t IntBits, std::size_t FracBits, bool IsSigned>
struct fix: public basic {
  static constexpr std::size_t type_size = IntBits + FracBits;

  static eda::hls::model::Type &type_info() {
    static auto type = eda::hls::model::FixedType(IntBits, FracBits, IsSigned);
    return type;
  } 

  static std::string type_name() {
    return "fix<" + std::to_string(IntBits)  + ","
                  + std::to_string(FracBits) + ","
                  + std::to_string(IsSigned) + ">";
  }
 
  std::string name() const override { return type_name(); }
  std::size_t size() const override { return type_size; }
  eda::hls::model::Type &info() const override { return type_info(); }
};

template<std::size_t IntBits, std::size_t FracBits>
using sfix = fix<IntBits, FracBits, true>;

template<std::size_t Bits>
using sint = sfix<Bits, 0>;

using sint8  = sint<8>;
using sint16 = sint<16>;
using sint32 = sint<32>;
using sint64 = sint<64>;

template<std::size_t IntBits, std::size_t FracBits>
using ufix = fix<IntBits, FracBits, false>;

template<std::size_t Bits>
using uint = ufix<Bits, 0>;

using bit    = uint<1>;
using uint8  = uint<8>;
using uint16 = uint<16>;
using uint32 = uint<32>;
using uint64 = uint<64>;

//===----------------------------------------------------------------------===//
// Float
//===----------------------------------------------------------------------===//

template<std::size_t ExpBits, std::size_t Precision>
struct real: public basic {
  static constexpr std::size_t type_size = ExpBits + Precision;

  static eda::hls::model::Type &type_info() {
    static auto type = eda::hls::model::FloatType(ExpBits, Precision);
    return type;
  }

  static std::string type_name() {
    return "real<" + std::to_string(ExpBits) + ","
                   + std::to_string(Precision) + ">";
  }

  std::string name() const override { return type_name(); }
  std::size_t size() const override { return type_size; }
  eda::hls::model::Type &info() const override { return type_info(); }
};

using float16 = real<5, 11>;
using float32 = real<8, 24>;
using float64 = real<11, 53>;

//===----------------------------------------------------------------------===//
// Bits
//===----------------------------------------------------------------------===//

template<std::size_t Bits>
struct bits: public basic {
  static constexpr std::size_t type_size = Bits;

  static eda::hls::model::Type &type_info() {
    static auto type = eda::hls::model::CustomType("bits");
    return type;
  }

  static std::string type_name() {
    return "bits<" + std::to_string(Bits) + ">";
  }

  std::string name() const override { return type_name(); }
  std::size_t size() const override { return type_size; }
  eda::hls::model::Type &info() const override { return type_info(); }
};

//===----------------------------------------------------------------------===//
// Composite
//===----------------------------------------------------------------------===//

class composite: public type {};

//===----------------------------------------------------------------------===//
// Tuple
//===----------------------------------------------------------------------===//

template<typename Head, typename... Tail>
struct tuple: public composite {
  static constexpr std::size_t type_size = Head::type_size + (... + Tail::type_size);

  static eda::hls::model::Type &type_info() {
    static std::vector<const eda::hls::model::Type*> types;
    types.push_back(&(Head::type_info()));
    ((types.push_back(&(Tail::type_info()))), ...);
    static auto type = eda::hls::model::TupleType(types);
    return type;
  }

  static std::string type_name() {
    std::stringstream out;
    out << "tuple<" << Head::type_name();
    ((out << "," << Tail::type_name()), ...);
    out << ">";
    return out.str();
  }

  std::string name() const override { return type_name(); }
  std::size_t size() const override { return type_size; }
  eda::hls::model::Type &info() const override { return type_info(); }
};

//===----------------------------------------------------------------------===//
// Complex
//===----------------------------------------------------------------------===//

template<typename Type>
struct complex: public tuple<Type, Type> {

  static std::string type_name() {
    return "complex<" + Type::type_name() + ">";
  }

  std::string name() const override { return type_name(); }
};

//===----------------------------------------------------------------------===//
// Tensor
//===----------------------------------------------------------------------===//

template<typename Type, std::size_t... Sizes>
struct tensor: public composite {
  static constexpr std::size_t type_size = Type::type_size * (... * Sizes);

  static eda::hls::model::Type &type_info() {
    static std::vector<std::size_t> sizes;
    ((sizes.push_back(Sizes)), ...);
    static auto type = eda::hls::model::TensorType(&(Type::type_info()), sizes);
    return type;
  }

  static std::string type_name() {
    std::stringstream out;
    out << "tensor<" << Type::type_name();
    ((out << "," << Sizes), ...);
    out << ">";
    return out.str();
  }

  std::string name() const override { return type_name(); }
  std::size_t size() const override { return type_size; }
  eda::hls::model::Type &info() const override { return type_info(); }
};

template<typename Type, std::size_t Size>
using vector = tensor<Type, Size>;

template<typename Type, std::size_t Rows, std::size_t Cols>
using matrix = tensor<Type, Rows, Cols>;

} // namespace dfc