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

#include "gate/flibrary.h"
#include "gate/netlist.h"

namespace eda {
namespace gate {

std::unique_ptr<FLibrary> FLibraryDefault::_instance;

bool FLibraryDefault::supports(FuncSymbol func) const {
  return true;
}

bool FLibraryDefault::synthesize(const Out &out, const std::vector<bool> &value, Netlist &net) {
  assert(out.size() == value.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    net.set_gate(out[i], (value[i] ? GateSymbol::ONE : GateSymbol::ZERO), {});
  }

  return true;
}

bool FLibraryDefault::synthesize(FuncSymbol func, const Out &out, const In &in, Netlist &net) {
  switch (func) {
  case FuncSymbol::NOP:
    return synth_unary_bitwise_op<GateSymbol::NOP>(out, in, net);
  case FuncSymbol::NOT:
    return synth_unary_bitwise_op<GateSymbol::NOT>(out, in, net);
  case FuncSymbol::AND:
    return synth_binary_bitwise_op<GateSymbol::AND>(out, in, net);
  case FuncSymbol::OR:
    return synth_binary_bitwise_op<GateSymbol::OR>(out, in, net);
  case FuncSymbol::XOR:
    return synth_binary_bitwise_op<GateSymbol::XOR>(out, in, net);
  case FuncSymbol::ADD:
    return synth_add(out, in, net);
  case FuncSymbol::SUB:
    return synth_sub(out, in, net);
  case FuncSymbol::MUX:
    return synth_mux(out, in, net);
  default:
    assert(false);
    return false;
  }
}

bool FLibraryDefault::synthesize(
    const Out &out, const In &in, const Control &control, Netlist &net) {
  assert(control.size() == 1 || control.size() == 2);
  assert(control.size() == in.size());

  Signal clock = invert_if_negative(control[0], net);
  if (control.size() == 1) {
    for (std::size_t i = 0; i < out.size(); i++) {
      Signal d = net.always(in[0][i]); // stored data
      net.set_gate(out[i], (clock.edge() ? GateSymbol::DFF : GateSymbol::LATCH), { d, clock });
    }
  } else {
    Signal edged = invert_if_negative(control[1], net);
    Signal reset = Signal::always(edged.gate());

    for (std::size_t i = 0; i < out.size(); i++) {
      Signal d = net.always(in[0][i]); // stored data
      Signal v = net.always(in[1][i]); // reset value
      Signal n = net.always(net.add_gate(GateSymbol::NOT, { v }));
      Signal r = net.level1(net.add_gate(GateSymbol::AND, { n, reset }));
      Signal s = net.level1(net.add_gate(GateSymbol::AND, { v, reset }));
      net.set_gate(out[i], GateSymbol::DFFrs, { d, clock, r, s });
    }
  }

  return true;
}

bool FLibraryDefault::synth_add(const Out &out, const In &in, Netlist &net) {
  // TODO:
  return synth_binary_bitwise_op<GateSymbol::AND>(out, in, net);
}

bool FLibraryDefault::synth_sub(const Out &out, const In &in, Netlist &net) {
  // TODO:
  return synth_binary_bitwise_op<GateSymbol::OR>(out, in, net);
}

bool FLibraryDefault::synth_mux(const Out &out, const In &in, Netlist &net) {
  assert(in.size() >= 4 && (in.size() & 1) == 0);
  const std::size_t n = in.size() / 2;

  for (std::size_t i = 0; i < out.size(); i++) {
    std::vector<Signal> temp;
    temp.reserve(n);

    for (std::size_t j = 0; j < n; j++) {
      const Arg &c = in[j];
      const Arg &x = in[j + n];
      assert(c.size() == 1 && out.size() == x.size());

      Signal cj0 = net.always(c[0]);
      Signal xji = net.always(x[i]);
      unsigned id = net.add_gate(GateSymbol::AND, { cj0, xji });

      temp.push_back(net.always(id));
    }

    net.set_gate(out[i], GateSymbol::OR, temp);
  }
  
  return true;
}

Signal FLibraryDefault::invert_if_negative(
    const std::pair<Event::Kind, unsigned> &entry, Netlist &net) {
  switch (entry.first) {
  case Event::POSEDGE:
    // Leave the clock signal unchanged.
    return net.posedge(entry.second);
  case Event::NEGEDGE:
    // Invert the clock signal.
    return net.posedge(net.add_gate(GateSymbol::NOT, { net.always(entry.second) }));
  case Event::LEVEL0:
    // Invert the enable signal.
    return net.level1(net.add_gate(GateSymbol::NOT, { net.always(entry.second) }));
  case Event::LEVEL1:
    // Leave the enable signal unchanged.
    return net.level1(entry.second);
  default:
    assert(false);
    return net.posedge(-1);
  }
}

}} // namespace eda::gate

