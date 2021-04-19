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

bool FLibraryDefault::synthesize(FuncSymbol func, const Out &out, const In &in, Netlist &net) {
  switch (func) {
  case FuncSymbol::NOP:
    return synthesize_nop(out, in, net);
  case FuncSymbol::NOT:
    return synthesize_not(out, in, net);
  case FuncSymbol::AND:
    return synthesize_and(out, in, net);
  case FuncSymbol::ADD:
    return synthesize_add(out, in, net);
  case FuncSymbol::SUB:
    return synthesize_sub(out, in, net);
  case FuncSymbol::MUX:
    return synthesize_mux(out, in, net);
  default:
    assert(false);
    return false;
  }
}

bool FLibraryDefault::synthesize_nop(const Out &out, const In &in, Netlist &net) {
  assert(in.size() == 1);

  const Arg &x = in[0];
  assert(out.size() == x.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    Signal xi = net.always(x[i]);
    net.set_gate(out[i], GateSymbol::NOP, { xi });
  }

  return true;
}

bool FLibraryDefault::synthesize_not(const Out &out, const In &in, Netlist &net) {
  assert(in.size() == 1);

  const Arg &x = in[0];
  assert(out.size() == x.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    Signal xi = net.always(x[i]);
    net.set_gate(out[i], GateSymbol::NOT, { xi });
  }

  return true;
}

bool FLibraryDefault::synthesize_and(const Out &out, const In &in, Netlist &net) {
  assert(in.size() == 2);

  const Arg &x = in[0];
  const Arg &y = in[1];
  assert(x.size() == y.size() && out.size() == x.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    Signal xi = net.always(x[i]);
    Signal yi = net.always(y[i]);
    net.set_gate(out[i], GateSymbol::AND, { xi, yi });
  }

  return true;
}

bool FLibraryDefault::synthesize_add(const Out &out, const In &in, Netlist &net) {
  // TODO:
  return synthesize_and(out, in, net);
}

bool FLibraryDefault::synthesize_sub(const Out &out, const In &in, Netlist &net) {
  // TODO:
  return synthesize_and(out, in, net);
}

bool FLibraryDefault::synthesize_mux(const Out &out, const In &in, Netlist &net) {
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

}} // namespace eda::gate

