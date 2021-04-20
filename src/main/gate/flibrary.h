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
#include <memory>
#include <vector>

#include "gate/gsymbol.h"
#include "gate/netlist.h"
#include "gate/signal.h"
#include "rtl/fsymbol.h"

using namespace eda::rtl;

namespace eda {
namespace gate {

class Netlist;

/**
 * \brief Interface for functional library.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>.
 */
struct FLibrary {
  typedef std::vector<unsigned> Arg;
  typedef Arg Out;
  typedef std::vector<Arg> In;
  typedef std::vector<std::pair<Event::Kind, unsigned>> Control;

  /// Checks if the library supports the given function.
  virtual bool supports(FuncSymbol func) const = 0;

  /// Synthesize the netlist for the given value.
  virtual bool synthesize(const Out &out, const std::vector<bool> &value, Netlist &net) = 0;
  /// Synthesize the netlist for the given function.
  virtual bool synthesize(FuncSymbol func, const Out &out, const In &in, Netlist &net) = 0;
  /// Synthesize the netlist for the given register.
  virtual bool synthesize(const Out &out, const In &in, const Control &control, Netlist &net) = 0;

  virtual ~FLibrary() {} 
};

class FLibraryDefault final: public FLibrary {
public:
  static FLibrary& instance() {
    if (_instance == nullptr) {
      _instance = std::unique_ptr<FLibrary>(new FLibraryDefault());
    }
    return *_instance;
  }

  bool supports(FuncSymbol func) const override;
  bool synthesize(const Out &out, const std::vector<bool> &value, Netlist &net) override;
  bool synthesize(FuncSymbol func, const Out &out, const In &in, Netlist &net) override;
  bool synthesize(const Out &out, const In &in, const Control &control, Netlist &net) override;

private:
  FLibraryDefault() {}
  ~FLibraryDefault() override {}

  bool synth_add(const Out &out, const In &in, Netlist &net);
  bool synth_sub(const Out &out, const In &in, Netlist &net);
  bool synth_mux(const Out &out, const In &in, Netlist &net);

  Signal invert_if_negative(const std::pair<Event::Kind, unsigned> &entry, Netlist &net);

  template<GateSymbol G> bool synth_unary_bitwise_op (const Out &out, const In &in, Netlist &net);
  template<GateSymbol G> bool synth_binary_bitwise_op(const Out &out, const In &in, Netlist &net);

  static std::unique_ptr<FLibrary> _instance;
};

template<GateSymbol G>
bool FLibraryDefault::synth_unary_bitwise_op(const Out &out, const In &in, Netlist &net) {
  assert(in.size() == 1);

  const Arg &x = in[0];
  assert(out.size() == x.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    Signal xi = net.always(x[i]);
    net.set_gate(out[i], G, { xi });
  }

  return true;
}

template<GateSymbol G>
bool FLibraryDefault::synth_binary_bitwise_op(const Out &out, const In &in, Netlist &net) {
  assert(in.size() == 2);

  const Arg &x = in[0];
  const Arg &y = in[1];
  assert(x.size() == y.size() && out.size() == x.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    Signal xi = net.always(x[i]);
    Signal yi = net.always(y[i]);
    net.set_gate(out[i], G, { xi, yi });
  }

  return true;
}

}} // namespace eda::gate

