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

class wire {
public:
  virtual std::string type() const = 0;

  virtual ~wire() {}

  const std::string name;
  const wire_kind kind;
  const wire_direct direct;

protected:
  wire(const std::string &name, wire_kind kind, wire_direct direct):
    name(name), kind(kind), direct(direct) { declare(this); }
  wire(wire_kind kind, wire_direct direct):
    wire(eda::utils::unique_name("wire"), kind, direct) {}

  void declare(const wire *var);

  void connect(const wire *in, const wire *out);

  void connect(const std::string &op,
               const std::vector<const wire*> &in,
               const std::vector<const wire*> &out);

  void connect(const std::string &op,
               const wire *lhs,
               const wire *rhs,
               const wire *out) {
    connect(op, { lhs, rhs }, { out });
  }
};

template<typename Type>
struct typed: public wire {
  typed(wire_kind kind, wire_direct direct):
    wire(kind, direct) {}
  typed(const std::string &name, wire_kind kind, wire_direct direct):
    wire(name, kind, direct) {}

  std::string type() const override {
    return Type::type_name();
  }

  typed(const typed<Type> &rhs): wire(rhs.name, rhs.kind, rhs.direct) {
    this->connect(&rhs, this);
  }

  typed<Type>& operator=(const typed<Type> &rhs) {
    this->connect(&rhs, this);
    return *this;
  }
 
  typed<Type>& operator+(const typed<Type> &rhs) {
    auto &out = create();
    this->connect("ADD", this, &rhs, &out);
    return out;
  }

  typed<Type>& operator-(const typed<Type> &rhs) {
    auto &out = create();
    this->connect("SUB", this, &rhs, &out);
    return out;
  }

  typed<Type>& operator*(const typed<Type> &rhs) {
    auto &out = create();
    this->connect("MUL", this, &rhs, &out);
    return out;
  }

  typed<Type>& operator/(const typed<Type> &rhs) {
    auto &out = create();
    this->connect("DIV", this, &rhs, &out);
    return out;
  }

protected:
  virtual typed<Type>* new_wire() const {
    return new typed(kind, INOUT);
  }

  typed<Type>& create() {
    static std::vector<std::shared_ptr<typed<Type>>> storage;

    std::shared_ptr<typed<Type>> ptr(new_wire());
    storage.push_back(ptr);

    return *ptr;
  }
};

//===----------------------------------------------------------------------===//
// DFC variables
//===----------------------------------------------------------------------===//

template<typename Type, wire_kind Kind, wire_direct Direct>
struct var: public typed<Type> {
  var(): typed<Type>(Kind, Direct) {}
  var(const typed<Type> &rhs): typed<Type>(rhs) {}
  explicit var(const std::string &id): typed<Type>(id, Kind, Direct) {}
};

/// Input constraints.
template<typename Type, wire_kind Kind>
struct var<Type, Kind, INPUT>: public typed<Type> {
  var(): typed<Type>(Kind, INPUT) {}
  explicit var(const std::string &id): typed<Type>(id, Kind, INPUT) {}

  // Assignment to an input is prohibited.
  var(const typed<Type> &rhs) = delete;
  typed<Type>& operator=(const typed<Type> &rhs) = delete;
};

/// Output constraints.
template<typename Type, wire_kind Kind>
struct var<Type, Kind, OUTPUT>: public typed<Type> {
  var(): typed<Type>(Kind, OUTPUT) {}
  var(const typed<Type> &rhs): typed<Type>(rhs) {}
  explicit var(const std::string &id): typed<Type>(id, Kind, OUTPUT) {}

  // Reading from an output is prohibited.
  typed<Type>& operator+(const typed<Type> &rhs) = delete;
  typed<Type>& operator-(const typed<Type> &rhs) = delete; 
  typed<Type>& operator*(const typed<Type> &rhs) = delete;
  typed<Type>& operator/(const typed<Type> &rhs) = delete;
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
