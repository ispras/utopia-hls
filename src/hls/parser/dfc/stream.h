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

#include <string>
#include <memory>
#include <vector>

#define DFC_VAR(var, Type) \
  dfc::var<Type> var

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

class stream {
public:
  std::string name() const { return id; }
  virtual std::string type() const = 0;

  virtual bool is_input()  const = 0;
  virtual bool is_output() const = 0;
  virtual bool is_scalar() const = 0;
  virtual bool is_const()  const = 0;

  virtual ~stream() {}

protected:
  explicit stream(const std::string &id): id(id) { declare(this); }
  stream(): stream(eda::utils::unique_name("stream")) {}

  void declare(const stream *var);

  void connect(const stream *in, const stream *out);

  void connect(const std::string &op,
               const std::vector<const stream*> &in,
               const std::vector<const stream*> &out);

  void connect(const std::string &op,
               const stream *lhs,
               const stream *rhs,
               const stream *out) {
    connect(op, { lhs, rhs }, { out });
  }

  /// Stores created streams.
  void store(const std::shared_ptr<stream> &var) {
    static std::vector<std::shared_ptr<stream>> vars;
    vars.push_back(var);
  }

  const std::string id;
};

template<typename Type>
struct var: public stream {
  var() = default;
  explicit var(const std::string &id): stream(id) {}

  std::string type() const override {
    return Type::type_name();
  }

  bool is_input()  const override { return false; }
  bool is_output() const override { return false; }
  bool is_scalar() const override { return false; }
  bool is_const()  const override { return false; }

  var(const var<Type> &rhs): stream() {
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
  var<Type>& create() {
    auto ptr = std::make_shared<var<Type>>();
    store(ptr);
    return *ptr;
  }
};

template<typename Type>
struct input final: public var<Type> {
  input() = default;
  explicit input(const std::string &id): var<Type>(id) {}

  bool is_input() const override { return true; }

  // Assignment to an input is prohibited.
  input(const var<Type> &rhs) = delete;
  var<Type>& operator =(const var<Type> &rhs) = delete;
};

template<typename Type>
struct output final: public var<Type> {
  output() = default;
  explicit output(const std::string &id): var<Type>(id) {}

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
  bool is_scalar() const override { return true; }

  // TODO:
};

template<typename Type>
struct scalar_input final: public scalar<Type> {
  scalar_input() = default;
  explicit scalar_input(const std::string &id): scalar<Type>(id) {}

  bool is_input() const override { return true; }

  // Assignment to an input is prohibited.
  scalar_input(const scalar<Type> &rhs) = delete;
  scalar<Type>& operator =(const scalar<Type> &rhs) = delete;
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
  scalar<Type>& operator+(const scalar<Type> &rhs) = delete;
  scalar<Type>& operator-(const scalar<Type> &rhs) = delete;
  scalar<Type>& operator*(const scalar<Type> &rhs) = delete;
  scalar<Type>& operator/(const scalar<Type> &rhs) = delete;
};

template<typename Type>
struct value final: public scalar<Type> {
  bool is_const() const override { return true; }

  // Assignment to a value is prohibited.
  value(const var<Type> &rhs) = delete;
  var<Type>& operator =(const var<Type> &rhs) = delete;
};

} // namespace dfc
