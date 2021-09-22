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

#include <memory>
#include <string>
#include <vector>

#include "hls/model/model.h"

using namespace eda::hls::model;

namespace eda::hls::parser {

struct AstDecl final {
  AstDecl(Variable::Kind kind, Variable::Bind bind,
      const std::string &name, const std::string &type):
    kind(kind), bind(bind), name(name), type(type) {}

  AstDecl() = default;

  Variable::Kind kind;
  Variable::Bind bind;
  std::string name;
  std::string type;
};

struct AstAssign final {
  AstAssign(FuncSymbol func, const std::string &out, const std::vector<std::string> &in):
    func(func), out(out), in(in) {}

  AstAssign() = default;

  FuncSymbol func;
  std::string out;
  std::vector<std::string> in;
};

struct AstProc final {
  AstProc() = default;

  Event::Kind event;
  std::string signal;
  std::string guard;
  std::vector<AstAssign> action;
};

struct AstModel final {
  AstModel() = default;

  std::vector<AstDecl> decls;
  std::vector<AstProc> procs;
};

/**
 * \brief Helps to contruct the IR from source code.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Builder final {
public:
  static Builder& get() {
    if (_instance == nullptr) {
      _instance = std::unique_ptr<Builder>(new Builder());
    }
    return *_instance;
  }

  std::vector<bool> to_value(const std::string &value) const;
  Variable to_var(const std::string &value) const;

  Type to_type(const std::string &type) const;

  std::unique_ptr<Net> create();

  void start_model() {
    _model.decls.clear();
    _model.procs.clear();
  }

  void add_decl(Variable::Kind kind, Variable::Bind bind,
      const std::string &name, const std::string &type) {
    _model.decls.push_back(AstDecl(kind, bind, name, type));
  }

  void start_proc() {
    _proc.event = Event::ALWAYS;
    _proc.signal.clear();
    _proc.guard.clear();
    _proc.action.clear();
  }

  void end_proc() {
    _model.procs.push_back(_proc);
  }

  void set_event(Event::Kind event, const std::string &signal) {
    _proc.event = event;
    _proc.signal = signal;
  }

  void set_guard(const std::string &guard) {
    _proc.guard = guard;
  }

  void add_assign(FuncSymbol func, const std::string &out, const std::vector<std::string> &in) {
    _proc.action.push_back(AstAssign(func, out, in));
  }

private:
  Builder() {}

  AstModel _model;
  AstProc _proc;

  static std::unique_ptr<Builder> _instance;
};

} // namespace eda::hls::parser
