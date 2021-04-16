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

#include "gate/netlist.h"
#include "rtl/net.h"
#include "rtl/vnode.h"
#include "utils/utils.h"

using namespace eda::rtl;
using namespace eda::utils;

namespace eda {
namespace gate {

unsigned Netlist::gate_id(const VNode *vnode) {
  const auto i = _gates_id.find(vnode->name());
  if (i != _gates_id.end()) {
    return i->second;
  }

  _gates_id.insert({ vnode->name(), _gates.size() });
  return _gates.size();
}

void Netlist::allocate_gates(const VNode *vnode) {
  assert(vnode != nullptr);

  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();

  for (unsigned i = 0; i < size; i++) {
    assert(base + i == _gates.size());
    _gates.push_back(new Gate(base + i));
  }
}

void Netlist::handle_src(const VNode *vnode) {
  // Do nothing.
}

void Netlist::handle_fun(const VNode *vnode) {
  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();

  // TODO: Synthesize the functional unit.
  for (unsigned i = 0; i < size; i++) {
    Gate *gate = _gates[base + i];

    std::vector<Signal> inputs;
    inputs.reserve(vnode->arity());

    for (size_t j = 0; j < vnode->arity(); j++) {
      const VNode *input = vnode->input(j);
      const unsigned input_base = gate_id(input);
      const unsigned input_size = input->var().type().width();

      inputs.push_back(Signal::always(_gates[(input_base + i) % input_size]));
    }

    // TODO: There should be corresponding function.
    gate->~Gate();
    new (gate) Gate(base + i, GateSymbol::AND, inputs);
  }
}

void Netlist::handle_mux(const VNode *vnode) {
  // TODO: Synthesize the multiplexor.
  handle_fun(vnode);
}

void Netlist::handle_reg(const VNode *vnode) {
  const Event &event = vnode->event();

  if (!event.edge()) {
    // TODO: Handle latches and asynchronous resets.
  }

  const VNode *clk = event.node();
  const unsigned clk_id = gate_id(clk);

  const unsigned base = gate_id(vnode);
  const unsigned size = vnode->var().type().width();

  // TODO: Synthesize the proper trigger.
  for (unsigned i = 0; i < size; i++) {
    Gate *gate = _gates[base + i];

    const VNode *data = vnode->input(0);
    const unsigned data_base = gate_id(data);

    // TODO: proper edge.
    gate->~Gate();
    new (gate) Gate(base + i, GateSymbol::DFF,
      { Signal::always(_gates[data_base + i]), Signal::posedge(_gates[clk_id]) } );
  }
}

void Netlist::create(const Net &net) {
  for (auto i = net.vbegin(); i != net.vend(); i++) {
    allocate_gates(*i);
  }

  for (auto i = net.vbegin(); i != net.vend(); i++) {
    switch ((*i)->kind()) {
    case VNode::SRC:
      handle_src(*i);
      break;
    case VNode::FUN:
      handle_fun(*i);
      break;
    case VNode::MUX:
      handle_mux(*i);
      break;
    case VNode::REG:
      handle_reg(*i);
      break;
    }
  }
}

std::ostream& operator <<(std::ostream &out, const Netlist &netlist) {
  for (auto i = netlist.begin(); i != netlist.end(); i++) {
    out << **i << std::endl;
  }

  return out;
}
 
}} // namespace eda::gate

