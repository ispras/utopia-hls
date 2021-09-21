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

#include <cassert>
#include <cstddef>

#include "gate/model/gate.h"
#include "gate/model/netlist.h"
#include "rtl/compiler/compiler.h"
#include "rtl/library/flibrary.h"
#include "rtl/model/net.h"
#include "rtl/model/vnode.h"
#include "util/utils.h"

using namespace eda::gate::model;
using namespace eda::rtl::library;
using namespace eda::rtl::model;
using namespace eda::utils;

namespace eda::rtl::compiler {

void Compiler::compile(const Net &net) {
  assert(_netlist.size() == 0);

  for (const auto vnode: net.vnodes()) {
    alloc_gates(vnode);
  }

  for (const auto vnode: net.vnodes()) {
    switch (vnode->kind()) {
    case VNode::SRC:
      synth_src(vnode);
      break;
    case VNode::VAL:
      synth_val(vnode);
      break;
    case VNode::FUN:
      synth_fun(vnode);
      break;
    case VNode::MUX:
      synth_mux(vnode);
      break;
    case VNode::REG:
      synth_reg(vnode);
      break;
    }
  }
}

unsigned Compiler::gate_id(const VNode *vnode) {
  const auto i = _gates_id.find(vnode->name());
  if (i != _gates_id.end()) {
    return i->second;
  }

  _gates_id.insert({ vnode->name(), _netlist.size() });
  return _netlist.size();
}

void Compiler::alloc_gates(const VNode *vnode) {
  assert(vnode != nullptr);

  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();

  for (unsigned i = 0; i < size; i++) {
    assert(base + i == _netlist.size());
    _netlist.add_gate(new Gate(base + i));
  }
}

Netlist::Out Compiler::out(const VNode *vnode) {
  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();

  Netlist::Out out(size);
  for (unsigned i = 0; i < size; i++) {
    out[i] = base + i;
  }

  return out;
}

Netlist::In Compiler::in(const VNode *vnode) {
  Netlist::In in(vnode->arity());
  for (std::size_t i = 0; i < vnode->arity(); i++) {
    in[i] = out(vnode->input(i));
  }

  return in;
}

void Compiler::synth_src(const VNode *vnode) {
  // Do nothing.
}

void Compiler::synth_val(const VNode *vnode) {
  _library.synthesize(out(vnode), vnode->value(), _netlist);
}

void Compiler::synth_fun(const VNode *vnode) {
  assert(_library.supports(vnode->func()));
  _library.synthesize(vnode->func(), out(vnode), in(vnode), _netlist);
}

void Compiler::synth_mux(const VNode *vnode) {
  assert(_library.supports(FuncSymbol::MUX));
  _library.synthesize(FuncSymbol::MUX, out(vnode), in(vnode), _netlist);
}

void Compiler::synth_reg(const VNode *vnode) {
  // Level (latch), edge (flip-flop), or edge and level (flip-flop /w set/reset).
  assert(vnode->esize() == 1 || vnode->esize() == 2);

  Netlist::ControlList control;
  for (const auto &event: vnode->events()) {
    control.push_back({ event.kind(), gate_id(event.node()) });
  }

  _library.synthesize(out(vnode), in(vnode), control, _netlist);
}

} // namespace eda::rtl::compiler
