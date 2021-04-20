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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "rtl/event.h"
#include "rtl/fsymbol.h"
#include "rtl/variable.h"

namespace eda {
namespace rtl {

class Net;
class PNode;

/**
 * \brief Represents a v-node (v = variable), a functional or communication unit of the design.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class VNode final {
  // Creation.
  friend class Net;
  // Setting the parent p-node.
  friend class PNode;

public:
  typedef std::vector<VNode *> List;

  enum Kind {
    /// Source node (s-node): input wire x.
    SRC,
    /// Functional node (f-node): always_comb y <= f(x[0], ..., x[n-1]).
    FUN,
    /// Multiplexor node (m-node): always_comb y <= mux(x[0], ..., x[n-1]).
    MUX,
    /// Register node (r-node): always_ff @(edge) y <= x or always_latch if(level) y <= x.
    REG
  };

  Kind kind() const { return _kind; }

  const Variable& var() const { return _var; }
  const std::string& name() const { return _var.name(); }
  const Type& type() const { return _var.type(); }

  const std::size_t esize() const { return _events.size(); }
  const Event::List& events() const { return _events; }
  const Event& event(std::size_t i) const { return _events[i]; }

  FuncSymbol func() const { return _func; }
  std::size_t arity() const { return _inputs.size(); }

  const List& inputs() const { return _inputs; }
  const VNode* input(std::size_t i) const { return _inputs[i]; }
  VNode* input(std::size_t i) { return _inputs[i]; }

  const PNode* pnode() const { return _pnode; }

private:
  VNode(Kind kind, const Variable &var, const Event::List &events,
      FuncSymbol func, const List &inputs):
    _kind(kind), _var(var), _events(events), _func(func), _inputs(inputs), _pnode(nullptr) {
    assert(std::find(inputs.begin(), inputs.end(), nullptr) == inputs.end());
  }

  VNode *duplicate(const std::string &new_name) {
    Variable var(new_name, _var.kind(), _var.bind(), _var.type());
    return new VNode(_kind, var, _events, _func, _inputs);
  }

  void replace_with(Kind kind, const Variable &var, const Event::List &events,
      FuncSymbol func, const List &inputs) {
    this->~VNode();
    new (this) VNode(kind, var, events, func, inputs);
  }

  void set_pnode(const PNode *pnode) {
    assert(pnode != nullptr);
    _pnode = pnode;
  }

  const Kind _kind;
  const Variable _var;
  const Event::List _events;
  const FuncSymbol _func;
  const List _inputs;

  // Parent p-node (set on p-node creation).
  const PNode *_pnode;
};

std::ostream& operator <<(std::ostream &out, const VNode &vnode);

}} // namespace eda::rtl

