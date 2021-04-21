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

namespace eda::gate {

std::unique_ptr<FLibrary> FLibraryDefault::_instance;

bool FLibraryDefault::supports(FuncSymbol func) const {
  return true;
}

void FLibraryDefault::synthesize(const Out &out, const std::vector<bool> &value, Netlist &net) {
  assert(out.size() == value.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    net.set_gate(out[i], (value[i] ? GateSymbol::ONE : GateSymbol::ZERO), {});
  }
}

void FLibraryDefault::synthesize(FuncSymbol func, const Out &out, const In &in, Netlist &net) {
  switch (func) {
  case FuncSymbol::NOP:
    synth_unary_bitwise_op<GateSymbol::NOP>(out, in, net);
    break;
  case FuncSymbol::NOT:
    synth_unary_bitwise_op<GateSymbol::NOT>(out, in, net);
    break;
  case FuncSymbol::AND:
    synth_binary_bitwise_op<GateSymbol::AND>(out, in, net);
    break;
  case FuncSymbol::OR:
    synth_binary_bitwise_op<GateSymbol::OR>(out, in, net);
    break;
  case FuncSymbol::XOR:
    synth_binary_bitwise_op<GateSymbol::XOR>(out, in, net);
    break;
  case FuncSymbol::ADD:
    synth_add(out, in, net);
    break;
  case FuncSymbol::SUB:
    synth_sub(out, in, net);
    break;
  case FuncSymbol::MUX:
    synth_mux(out, in, net);
    break;
  default:
    assert(false);
  }
}

void FLibraryDefault::synthesize(
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
}

void FLibraryDefault::synth_add(const Out &out, const In &in, Netlist &net) {
  synth_adder(out, in, false, net);
}

void FLibraryDefault::synth_sub(const Out &out, const In &in, Netlist &net) {
  // The two's complement code: (x - y) == (x + ~y + 1).
  const Arg &x = in[0];
  const Arg &y = in[1];

  Out temp(y.size());
  for (std::size_t i = 0; i < y.size(); i++) {
    temp[i] = net.add_gate();
  }

  synth_unary_bitwise_op<GateSymbol::NOT>(temp, { y }, net);
  synth_adder(out, { x, temp }, true, net);
}

void FLibraryDefault::synth_adder(const Out &out, const In &in, bool plus_one, Netlist &net) {
  assert(in.size() == 2);

  const Arg &x = in[0];
  const Arg &y = in[1];
  assert(x.size() == y.size() && out.size() == x.size());

  unsigned c_in = -1;
  unsigned c_out = net.add_gate(plus_one ? GateSymbol::ONE : GateSymbol::ZERO, {});

  for (std::size_t i = 0; i < out.size(); i++) {
    c_in = c_out;
    c_out = (i != out.size() - 1 ? net.add_gate() : -1);
    synth_adder(out[i], c_out, x[i], y[i], c_in, net);
  }
}

void FLibraryDefault::synth_adder(unsigned z, unsigned c_out,
    unsigned x, unsigned y, unsigned c_in, Netlist &net) {
  Signal x_wire = net.always(x);
  Signal y_wire = net.always(y);
  Signal c_wire = net.always(c_in);

  // z = (x + y) + c_in (mod 2).
  Signal x_plus_y = net.always(net.add_gate(GateSymbol::XOR, { x_wire, y_wire }));
  net.set_gate(z, GateSymbol::XOR, { x_plus_y, c_wire });

  if (c_out != -1u) {
    // c_out = (x & y) | (x + y) & c_in.
    Signal x_and_y = net.always(net.add_gate(GateSymbol::AND, { x_wire, y_wire }));
    Signal x_plus_y_and_c_in = net.always(net.add_gate(GateSymbol::AND, { x_plus_y, c_wire }));
    net.set_gate(c_out, GateSymbol::OR, { x_and_y, x_plus_y_and_c_in });
  }
}

void FLibraryDefault::synth_mux(const Out &out, const In &in, Netlist &net) {
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

} // namespace eda::gate
