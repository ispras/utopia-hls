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

class wire {
public:
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

  void declare(const wire *var) const;

  void connect(const wire *in, const wire *out) const;

  void connect(const std::string &opcode,
               const std::vector<const wire*> &in,
               const std::vector<const wire*> &out) const;

  void connect(const std::string &op,
               const wire *in,
               const wire *out) const {
    std::vector source = { in };
    std::vector target = { out };
    connect(op, source, target);
  }

  void connect(const std::string &op,
               const wire *lhs,
               const wire *rhs,
               const wire *out) const {
    std::vector source = { lhs, rhs };
    std::vector target = { out };
    connect(op, source, target);
  }

  void store(wire *var) const {
    static std::vector<std::unique_ptr<wire>> storage;
    storage.push_back(std::unique_ptr<wire>(var));
  }
};

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
    auto &out = create();
    connect("ADD", this, &rhs, &out);
    return out;
  }

  typed<Type>& operator-(const typed<Type> &rhs) const {
    auto &out = create();
    connect("SUB", this, &rhs, &out);
    return out;
  }

  typed<Type>& operator*(const typed<Type> &rhs) const {
    auto &out = create();
    connect("MUL", this, &rhs, &out);
    return out;
  }

  typed<Type>& operator/(const typed<Type> &rhs) const {
    auto &out = create();
    connect("DIV", this, &rhs, &out);
    return out;
  }

  typed<Type>& operator>>(std::size_t rhs) const {
    auto &out = create();
    connect("SHR", this, /* FIXME: const(rhs) */ &out);
    return out;
  }

  typed<Type>& operator<<(std::size_t rhs) const {
    auto &out = create();
    connect("SHL", this, /* FIXME: const(rhs) */ &out);
    return out;
  }

  typed<Type>& operator+=(const typed<Type> &rhs) { return *this = *this + rhs; }
  typed<Type>& operator-=(const typed<Type> &rhs) { return *this = *this - rhs; }
  typed<Type>& operator*=(const typed<Type> &rhs) { return *this = *this * rhs; }
  typed<Type>& operator/=(const typed<Type> &rhs) { return *this = *this / rhs; }

  template<typename NewType>
  typed<NewType>& cast() const {
    auto &out = create<NewType>();
    connect("CAST", this, &out);
    return out;
  }

protected:
  template<typename NewType = Type>
  typed<NewType>& create(wire_direct direct = INOUT) const {
    auto *wire = new typed<NewType>(kind, direct);
    store(wire);
    return *wire;
  }
};

//===----------------------------------------------------------------------===//
// DFC variables
//===----------------------------------------------------------------------===//

template<typename Type, wire_kind Kind, wire_direct Direct>
struct var: public typed<Type> {
  var(): typed<Type>(Kind, Direct) {}
  explicit var(const typed<Type> &rhs): typed<Type>(rhs) {}
  explicit var(const std::string &id): typed<Type>(id, Kind, Direct) {}

  var<Type, Kind, INPUT>& to_input() const {
    // Using the same name makes this input identical to the original wire.
    auto *in = new var<Type, Kind, INPUT>(wire::name);
    typed<Type>::store(in);
    return *in;
  }

  var<Type, Kind, OUTPUT>& to_output() const {
    // Using the same name makes this output identical to the original wire.
    auto *out = new var<Type, Kind, OUTPUT>(wire::name);
    typed<Type>::store(out);
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
// DFC streams
//===----------------------------------------------------------------------===//

template<typename Type> using stream_inout  = var<Type, STREAM, INOUT>;
template<typename Type> using stream_input  = var<Type, STREAM, INPUT>;
template<typename Type> using stream_output = var<Type, STREAM, OUTPUT>;

//===----------------------------------------------------------------------===//
// DFC scalars
//===----------------------------------------------------------------------===//

template<typename Type> using scalar_inout  = var<Type, SCALAR, INOUT>;
template<typename Type> using scalar_input  = var<Type, SCALAR, INPUT>;
template<typename Type> using scalar_output = var<Type, SCALAR, OUTPUT>;

//===----------------------------------------------------------------------===//
// DFC constants
//===----------------------------------------------------------------------===//

template<typename Type> using value = var<Type, CONST, INPUT>;

//===----------------------------------------------------------------------===//
// DFC shortcuts
//===----------------------------------------------------------------------===//

template<typename Type> using stream = stream_inout<Type>;
template<typename Type> using scalar = scalar_inout<Type>;

template<typename Type> using input  = stream_input<Type>;
template<typename Type> using output = stream_output<Type>;

} // namespace dfc
