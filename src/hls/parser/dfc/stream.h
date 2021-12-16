//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/parser/dfc/types.h"
#include "util/string.h"

#include <memory>
#include <string>
#include <vector>

#define DFC_STREAM(var, Type) \
  dfc::stream<Type> var

#define DFC_INPUT(var, Type) \
  dfc::input<Type> var

#define DFC_OUTPUT(var, Type) \
  dfc::output<Type> var

#define DFC_SCALAR(var, Type) \
  dfc::scalar<Type> var

#define DFC_SCALAR_INPUT(var, Type) \
  dfc::scalar_input<Type> var

#define DFC_SCALAR_OUTPUT(var, Type) \
  dfc::scalar_input<Type> var

#define DFC_VALUE(var, Type) \
  dfc::value<Type> var

namespace dfc {

enum wire_kind { CONST, SCALAR, STREAM };

enum wire_direct { INPUT, OUTPUT, INOUT };

// TODO:
struct wire_value {
  wire_value() {}
  wire_value(int rhs) {}

  std::string to_string() const { return ""; }

  const std::string type;
  const std::size_t size = 0;
  const std::vector<bool> value;
};

//===----------------------------------------------------------------------===//
// Untyped Wire
//===----------------------------------------------------------------------===//

struct wire {
  virtual std::string type() const = 0;
  virtual ~wire() {}

  const std::string name;
  const wire_kind kind;
  const wire_direct direct;
  const wire_value value;

protected:
  wire(const std::string &name,
       wire_kind kind,
       wire_direct direct,
       wire_value value):
       name(name), kind(kind), direct(direct), value(value) {}

  wire(const std::string &name, wire_kind kind, wire_direct direct):
      wire(name, kind, direct, 0) {}
};

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

void declare(const wire *var);

void connect(const wire *in, const wire *out);

void connect(const std::string &opcode,
             const std::vector<const wire*> &in,
             const std::vector<const wire*> &out);

inline void store(wire *var) {
  static std::vector<std::unique_ptr<wire>> storage;
  storage.push_back(std::unique_ptr<wire>(var));
}

template <typename Type> struct typed;

template<typename Type>
typed<Type>& create(wire_kind kind, wire_direct direct = INOUT) {
  auto *wire = new typed<Type>(kind, direct);
  store(wire);
  return *wire;
}

//===----------------------------------------------------------------------===//
// Typed Wire
//===----------------------------------------------------------------------===//

template<typename Type>
struct typed: public wire {
  typed(const std::string &name, wire_kind kind, wire_direct direct):
    wire(name, kind, direct) { declare(this); }

  typed(wire_kind kind, wire_direct direct):
    typed(eda::utils::unique_name("wire"), kind, direct) {}

  std::string type() const override {
    return Type::type_name();
  }

  typed(const typed<Type> &rhs, wire_direct direct):
      wire(rhs.name, rhs.kind, direct) {
    // Using the same name makes this wire identical to RHS.
  }

  explicit typed(const typed<Type> &rhs): typed(rhs, INOUT) {}

  typed<Type>& operator=(const typed<Type> &rhs) {
    connect(&rhs, this);
    return *this;
  }
 
  typed<Type>& operator+(const typed<Type> &rhs) const {
    auto &out = create<Type>(kind);
    connect("ADD", { this, &rhs }, { &out });
    return out;
  }

  typed<Type>& operator-(const typed<Type> &rhs) const {
    auto &out = create<Type>(kind);
    connect("SUB", { this, &rhs }, { &out });
    return out;
  }

  typed<Type>& operator*(const typed<Type> &rhs) const {
    auto &out = create<Type>(kind);
    connect("MUL", { this, &rhs }, { &out });
    return out;
  }

  typed<Type>& operator/(const typed<Type> &rhs) const {
    auto &out = create<Type>(kind);
    connect("DIV", { this, &rhs }, { &out });
    return out;
  }

  typed<Type>& operator>>(std::size_t rhs) const {
    auto &out = create<Type>(kind);
    connect("SHR" + std::to_string(rhs), { this }, { &out });
    return out;
  }

  typed<Type>& operator<<(std::size_t rhs) const {
    auto &out = create<Type>(kind);
    connect("SHL" + std::to_string(rhs), { this }, { &out });
    return out;
  }

  typed<bit>& operator==(const typed<Type> &rhs) const {
    auto &out = create<bit>(kind);
    connect("EQ", { this, &rhs }, { &out });
    return out;
  }

  typed<bit>& operator!=(const typed<Type> &rhs) const {
    auto &out = create<bit>(kind);
    connect("NE", { this, &rhs }, { &out });
    return out;
  }

  typed<bit>& operator>(const typed<Type> &rhs) const {
    auto &out = create<bit>(kind);
    connect("GT", { this, &rhs }, { &out });
    return out;
  }

  typed<bit>& operator>=(const typed<Type> &rhs) const {
    auto &out = create<bit>(kind);
    connect("GE", { this, &rhs }, { &out });
    return out;
  }

  typed<bit>& operator<(const typed<Type> &rhs) const {
    auto &out = create<bit>(kind);
    connect("LT", { this, &rhs }, { &out });
    return out;
  }

  typed<bit>& operator<=(const typed<Type> &rhs) const {
    auto &out = create<bit>(kind);
    connect("LE", { this, &rhs }, { &out });
    return out;
  }

  typed<Type>& operator+=(const typed<Type> &rhs) { return *this = *this + rhs; }
  typed<Type>& operator-=(const typed<Type> &rhs) { return *this = *this - rhs; }
  typed<Type>& operator*=(const typed<Type> &rhs) { return *this = *this * rhs; }
  typed<Type>& operator/=(const typed<Type> &rhs) { return *this = *this / rhs; }

  template<typename NewType>
  typed<NewType>& cast() const {
    auto &out = create<NewType>(kind);
    connect("CAST", { this }, { &out });
    return out;
  }
};

template<typename Type>
typed<Type>& mux(const typed<bit>  &sel,
                 const typed<Type> &lhs,
                 const typed<Type> &rhs) {
  auto &out = create<Type>(lhs.kind);
  connect("MUX", { &sel, &lhs, &rhs }, { &out });
  return out;
}

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

template<typename Type, wire_kind Kind, wire_direct Direct>
struct var: public typed<Type> {
  var(): typed<Type>(Kind, Direct) {}
  explicit var(const typed<Type> &rhs): typed<Type>(rhs) {}
  explicit var(const std::string &id): typed<Type>(id, Kind, Direct) {}

  var<Type, Kind, INPUT>& to_input() const {
    // Using the same name makes this input identical to the original wire.
    auto *in = new var<Type, Kind, INPUT>(wire::name);
    store(in);
    return *in;
  }

  var<Type, Kind, OUTPUT>& to_output() const {
    // Using the same name makes this output identical to the original wire.
    auto *out = new var<Type, Kind, OUTPUT>(wire::name);
    store(out);
    return *out;
  }

  typed<Type>& operator=(const typed<Type> &rhs) {
    return typed<Type>::operator=(rhs);
  }
};

/// Input specialization.
template<typename Type, wire_kind Kind>
struct var<Type, Kind, INPUT>: public typed<Type> {
  var(): typed<Type>(Kind, INPUT) {}
  explicit var(const typed<Type> &rhs): typed<Type>(rhs, INPUT) {}
  explicit var(const std::string &id): typed<Type>(id, Kind, INPUT) {}

  // Assignment to an input is prohibited.
  typed<Type>& operator=(const typed<Type> &rhs) = delete;
};

/// Output specialization.
template<typename Type, wire_kind Kind>
struct var<Type, Kind, OUTPUT>: public typed<Type> {
  var(): typed<Type>(Kind, OUTPUT) {}
  explicit var(const typed<Type> &rhs): typed<Type>(rhs, OUTPUT) {}
  explicit var(const std::string &id): typed<Type>(id, Kind, OUTPUT) {}

  typed<Type>& operator=(const typed<Type> &rhs) {
    return typed<Type>::operator=(rhs);
  }

  // Reading from an output is prohibited.
  typed<Type>& operator+(const typed<Type> &rhs) = delete;
  typed<Type>& operator-(const typed<Type> &rhs) = delete; 
  typed<Type>& operator*(const typed<Type> &rhs) = delete;
  typed<Type>& operator/(const typed<Type> &rhs) = delete;
};

/// Constant specialization.
template<typename Type>
struct var<Type, CONST, INPUT>: public typed<Type> {
  var(): typed<Type>(CONST, INPUT) {}
  explicit var(const std::string &id): typed<Type>(id, CONST, INPUT) {}

  // Assignment to a constant is prohibited.
  var(const typed<Type> &rhs) = delete;
  typed<Type>& operator=(const typed<Type> &rhs) = delete;

  // Initialization from a literal.
  var(wire_value value):
    typed<Type>(Type::type_name() + value.to_string(), CONST, INPUT) {}

  template<typename LiteralType>
  var(LiteralType value): var(wire_value(value)) {}
};

//===----------------------------------------------------------------------===//
// Streams
//===----------------------------------------------------------------------===//

template<typename Type> using stream_inout  = var<Type, STREAM, INOUT>;
template<typename Type> using stream_input  = var<Type, STREAM, INPUT>;
template<typename Type> using stream_output = var<Type, STREAM, OUTPUT>;

//===----------------------------------------------------------------------===//
// Scalars
//===----------------------------------------------------------------------===//

template<typename Type> using scalar_inout  = var<Type, SCALAR, INOUT>;
template<typename Type> using scalar_input  = var<Type, SCALAR, INPUT>;
template<typename Type> using scalar_output = var<Type, SCALAR, OUTPUT>;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

template<typename Type> using value = var<Type, CONST, INPUT>;

//===----------------------------------------------------------------------===//
// Shortcuts
//===----------------------------------------------------------------------===//

template<typename Type> using stream = stream_inout<Type>;
template<typename Type> using scalar = scalar_inout<Type>;

template<typename Type> using input  = stream_input<Type>;
template<typename Type> using output = stream_output<Type>;

} // namespace dfc
