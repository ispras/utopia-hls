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

#include "gate/flibrary.h"
#include "gate/netlist.h"
#include "rtl/net.h"
#include "rtl/vnode.h"
#include "utils/utils.h"

using namespace eda::rtl;
using namespace eda::utils;

namespace eda::gate {

void Netlist::create(const Net &net, FLibrary &lib) {
  for (const auto vnode: net.vnodes()) {
    alloc_gates(vnode);
  }

  for (const auto vnode: net.vnodes()) {
    switch (vnode->kind()) {
    case VNode::SRC:
      synth_src(vnode, lib);
      break;
    case VNode::VAL:
      synth_val(vnode, lib);
      break;
    case VNode::FUN:
      synth_fun(vnode, lib);
      break;
    case VNode::MUX:
      synth_mux(vnode, lib);
      break;
    case VNode::REG:
      synth_reg(vnode, lib);
      break;
    }
  }
}

unsigned Netlist::gate_id(const VNode *vnode) {
  const auto i = _gates_id.find(vnode->name());
  if (i != _gates_id.end()) {
    return i->second;
  }

  _gates_id.insert({ vnode->name(), _gates.size() });
  return _gates.size();
}

void Netlist::alloc_gates(const VNode *vnode) {
  assert(vnode != nullptr);

  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();

  for (unsigned i = 0; i < size; i++) {
    assert(base + i == _gates.size());
    _gates.push_back(new Gate(base + i));
  }
}

Netlist::Out Netlist::out(const VNode *vnode) {
  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();

  Out out(size);
  for (unsigned i = 0; i < size; i++) {
    out[i] = base + i;
  }

  return out;
}

Netlist::In Netlist::in(const VNode *vnode) {
  In in(vnode->arity());
  for (std::size_t i = 0; i < vnode->arity(); i++) {
    in[i] = out(vnode->input(i));
  }

  return in;
}

void Netlist::synth_src(const VNode *vnode, FLibrary &lib) {
  // Do nothing.
}

void Netlist::synth_val(const VNode *vnode, FLibrary &lib) {
  lib.synthesize(out(vnode), vnode->value(), *this);
}

void Netlist::synth_fun(const VNode *vnode, FLibrary &lib) {
  assert(lib.supports(vnode->func()));
  lib.synthesize(vnode->func(), out(vnode), in(vnode), *this);
}

void Netlist::synth_mux(const VNode *vnode, FLibrary &lib) {
  assert(lib.supports(FuncSymbol::MUX));
  lib.synthesize(FuncSymbol::MUX, out(vnode), in(vnode), *this);
}

void Netlist::synth_reg(const VNode *vnode, FLibrary &lib) {
  // Level (latch), edge (flip-flop), or edge and level (flip-flop /w set/reset).
  assert(vnode->esize() == 1 || vnode->esize() == 2);

  ControlList control;
  for (const auto &event: vnode->events()) {
    control.push_back({ event.kind(), gate_id(event.node()) });
  }

  lib.synthesize(out(vnode), in(vnode), control, *this);
}

std::ostream& operator <<(std::ostream &out, const Netlist &netlist) {
  for (const auto gate: netlist.gates()) {
    out << *gate << std::endl;
  }
  return out;
}
 
} // namespace eda::gate
