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
#include <memory>

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

std::unique_ptr<Netlist> Compiler::compile(const Net &net) {
  std::unique_ptr<Netlist> netlist = std::make_unique<Netlist>();

  _gates_id.clear();

  for (const auto vnode: net.vnodes()) {
    alloc_gates(vnode, *netlist);
  }

  for (const auto vnode: net.vnodes()) {
    switch (vnode->kind()) {
    case VNode::SRC:
      synth_src(vnode, *netlist);
      break;
    case VNode::VAL:
      synth_val(vnode, *netlist);
      break;
    case VNode::FUN:
      synth_fun(vnode, *netlist);
      break;
    case VNode::MUX:
      synth_mux(vnode, *netlist);
      break;
    case VNode::REG:
      synth_reg(vnode, *netlist);
      break;
    }
  }

  return netlist;
}

unsigned Compiler::gate_id(const VNode *vnode) const {
  const auto i = _gates_id.find(vnode->name());
  if (i != _gates_id.end()) {
    return i->second;
  }

  return -1u;
}

unsigned Compiler::gate_id(const VNode *vnode, const Netlist &netlist) {
  const unsigned id = gate_id(vnode);
  if (id != -1u) {
    return id;
  }

  _gates_id.insert({ vnode->name(), netlist.size() });
  return netlist.size();
}

void Compiler::alloc_gates(const VNode *vnode, Netlist &netlist) {
  assert(vnode != nullptr);

  const unsigned base = gate_id(vnode, netlist);
  const unsigned size = vnode->var().type().width();

  for (unsigned i = 0; i < size; i++) {
    assert(base + i == netlist.size());
    netlist.add_gate(new Gate(base + i));
  }
}

void Compiler::synth_src(const VNode *vnode, Netlist &netlist) {
  // Do nothing.
}

void Compiler::synth_val(const VNode *vnode, Netlist &netlist) {
  _library.synthesize(out(vnode), vnode->value(), netlist);
}

void Compiler::synth_fun(const VNode *vnode, Netlist &netlist) {
  assert(_library.supports(vnode->func()));
  _library.synthesize(vnode->func(), out(vnode), in(vnode), netlist);
}

void Compiler::synth_mux(const VNode *vnode, Netlist &netlist) {
  assert(_library.supports(FuncSymbol::MUX));
  _library.synthesize(FuncSymbol::MUX, out(vnode), in(vnode), netlist);
}

void Compiler::synth_reg(const VNode *vnode, Netlist &netlist) {
  // Level (latch), edge (flip-flop), or edge and level (flip-flop /w set/reset).
  assert(vnode->esize() == 1 || vnode->esize() == 2);

  Netlist::ControlList control;
  for (const auto &event: vnode->events()) {
    control.push_back({ event.kind(), gate_id(event.node()) });
  }

  _library.synthesize(out(vnode), in(vnode), control, netlist);
}

Netlist::In Compiler::in(const VNode *vnode) {
  Netlist::In in(vnode->arity());
  for (std::size_t i = 0; i < vnode->arity(); i++) {
    in[i] = out(vnode->input(i));
  }

  return in;
}

Netlist::Out Compiler::out(const VNode *vnode) {
  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();
  assert(base != -1u);

  Netlist::Out out(size);
  for (unsigned i = 0; i < size; i++) {
    out[i] = base + i;
  }

  return out;
}

} // namespace eda::rtl::compiler
