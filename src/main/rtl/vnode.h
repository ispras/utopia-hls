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
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const VNode &vnode);

public:
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

  const PNode* pnode() const { return _pnode; }

  const std::string& name() const { return _var.name(); }
  Kind kind() const { return _kind; }
  const Variable &var() const { return _var; }
  const Event& event() const { return _event; }
  FuncSymbol func() const { return _func; }
  std::size_t arity() const { return _inputs.size(); }
  const VNode* input(size_t i) const { return _inputs[i]; }

private:
  VNode(Kind kind, const Variable &var, const Event &event, FuncSymbol func,
      const std::vector<VNode *> &inputs):
    _pnode(nullptr), _kind(kind), _var(var), _event(event), _func(func), _inputs(inputs) {}

  VNode *duplicate(const std::string &new_name) {
    Variable var(new_name, _var.kind(), _var.bind(), _var.type());
    return new VNode(_kind, var, _event, _func, _inputs);
  }

  void replace_with(Kind kind, const Variable &var, const Event &event, FuncSymbol func,
      const std::vector<VNode *> &inputs) {
    this->~VNode();
    new (this) VNode(kind, var, event, func, inputs);
  }

  void set_pnode(PNode *pnode) {
    assert(pnode != nullptr);
    _pnode = pnode;
  }

  // Parent p-node (set on p-node creation).
  PNode *_pnode;

  const Kind _kind;
  const Variable _var;
  const Event _event;
  const FuncSymbol _func;
  const std::vector<VNode *> _inputs;
};

std::ostream& operator <<(std::ostream &out, const VNode &vnode);

}} // namespace eda::rtl

