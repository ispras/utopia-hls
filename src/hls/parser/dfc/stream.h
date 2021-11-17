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

#define DFC_VAR(var, Type) \
  dfc::var<Type> var

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

class wire {
public:
  std::string name() const { return id; }
  virtual std::string type() const = 0;

  virtual bool is_input()  const = 0;
  virtual bool is_output() const = 0;
  virtual bool is_scalar() const = 0;
  virtual bool is_const()  const = 0;

  virtual ~wire() {}

protected:
  explicit wire(const std::string &id): id(id) {
    declare(this);
  }

  wire(): wire(eda::utils::unique_name("wire")) {}

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

  /// Stores created wires.
  void store(const std::shared_ptr<wire> &w) {
    static std::vector<std::shared_ptr<wire>> storage;
    storage.push_back(w);
  }

  const std::string id;
};

template<typename Type>
struct var: public wire {
  var(bool scalar = false): wire(), scalar(scalar) {}
  var(const std::string &id, bool scalar = false): wire(id) {}

  std::string type() const override {
    return Type::type_name();
  }

  bool is_input()  const override { return false; }
  bool is_output() const override { return false; }
  bool is_scalar() const override { return scalar; }
  bool is_const()  const override { return false; }

  var(const var<Type> &rhs): wire(), scalar(rhs.scalar) {
    this->connect(&rhs, this);
  }

  var<Type>& operator=(const var<Type> &rhs) {
    this->connect(&rhs, this);
    return *this;
  }
 
  var<Type>& operator+(const var<Type> &rhs) {
    auto &out = create();
    this->connect("ADD", this, &rhs, &out);
    return out;
  }

  var<Type>& operator-(const var<Type> &rhs) {
    auto &out = create();
    this->connect("SUB", this, &rhs, &out);
    return out;
  }

  var<Type>& operator*(const var<Type> &rhs) {
    auto &out = create();
    this->connect("MUL", this, &rhs, &out);
    return out;
  }

  var<Type>& operator/(const var<Type> &rhs) {
    auto &out = create();
    this->connect("DIV", this, &rhs, &out);
    return out;
  }

private:
  /// Creates a variable (and stores the pointer to avoid memory leak).
  var<Type>& create() { // FIXME: Scalar or stream
    auto ptr = std::make_shared<var<Type>>();
    store(ptr);
    return *ptr;
  }

  const bool scalar;
};

template<typename Type>
struct stream: public var<Type> {
  stream(): var<Type>(true) {}
  explicit stream(const std::string &id): var<Type>(id, true) {}
};

template<typename Type>
struct input final: public stream<Type> {
  input() = default;
  explicit input(const std::string &id): stream<Type>(id) {}

  bool is_input() const override { return true; }

  // Assignment to an input is prohibited.
  input(const var<Type> &rhs) = delete;
  var<Type>& operator =(const var<Type> &rhs) = delete;
};

template<typename Type>
struct output final: public stream<Type> {
  output() = default;
  explicit output(const std::string &id): stream<Type>(id) {}

  bool is_output() const override { return true; }

  output(const var<Type> &rhs): var<Type>() {
    this->connect(&rhs, this);
  }
  
  output<Type>& operator =(const var<Type> &rhs) {
    this->connect(&rhs, this);
    return *this;
  }

  // Reading from an output is prohibited.
  var<Type>& operator+(const var<Type> &rhs) = delete;
  var<Type>& operator-(const var<Type> &rhs) = delete; 
  var<Type>& operator*(const var<Type> &rhs) = delete;
  var<Type>& operator/(const var<Type> &rhs) = delete;
};

template<typename Type>
struct scalar: public var<Type> {
  scalar(): var<Type>(false) {}
  explicit scalar(const std::string &id): var<Type>(id, false) {}
};

template<typename Type>
struct scalar_input final: public scalar<Type> {
  scalar_input() = default;
  explicit scalar_input(const std::string &id): scalar<Type>(id) {}

  bool is_input() const override { return true; }

  // Assignment to an input is prohibited.
  scalar_input(const var<Type> &rhs) = delete;
  var<Type>& operator =(const var<Type> &rhs) = delete;
};

template<typename Type>
struct scalar_output final: public scalar<Type> {
  scalar_output() = default;
  explicit scalar_output(const std::string &id): scalar<Type>(id) {}

  bool is_output() const override { return true; }

  scalar_output(const scalar<Type> &rhs): scalar<Type>() {
    this->connect(&rhs, this);
  }
 
  scalar_output<Type>& operator =(const scalar<Type> &rhs) {
    this->connect(&rhs, this);
    return *this;
  }

  // Reading from an output is prohibited.
  var<Type>& operator+(const var<Type> &rhs) = delete;
  var<Type>& operator-(const var<Type> &rhs) = delete;
  var<Type>& operator*(const var<Type> &rhs) = delete;
  var<Type>& operator/(const var<Type> &rhs) = delete;
};

template<typename Type>
struct value final: public scalar<Type> {
  // TODO:
  value() = default;
  explicit value(const std::string &id): scalar<Type>(id) {}

  bool is_const() const override { return true; }

  // Assignment to a value is prohibited.
  value(const var<Type> &rhs) = delete;
  var<Type>& operator =(const var<Type> &rhs) = delete;
};

} // namespace dfc
