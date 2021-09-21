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

#include "gate/model/gsymbol.h"
#include "gate/model/netlist.h"
#include "gate/model/signal.h"
#include "rtl/model/fsymbol.h"

using namespace eda::gate::model;
using namespace eda::rtl::model;

namespace eda::gate::model {
  class Netlist;
} // namespace eda::gate::model

namespace eda::rtl::library {

/**
 * \brief Interface for functional library.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>.
 */
struct FLibrary {
  using GateIdList = Netlist::GateIdList;
  using Value = Netlist::Value;
  using In = Netlist::In;
  using Out = Netlist::Out;
  using ControlEvent = Netlist::ControlEvent;
  using ControlList = Netlist::ControlList;

  /// Checks if the library supports the given function.
  virtual bool supports(FuncSymbol func) const = 0;

  /// Synthesize the netlist for the given value.
  virtual void synthesize(
    const Out &out, const Value &value, Netlist &net) = 0;
  /// Synthesize the netlist for the given function.
  virtual void synthesize(
    FuncSymbol func, const Out &out, const In &in, Netlist &net) = 0;
  /// Synthesize the netlist for the given register.
  virtual void synthesize(
    const Out &out, const In &in, const ControlList &control, Netlist &net) = 0;

  virtual ~FLibrary() {} 
};

class FLibraryDefault final: public FLibrary {
public:
  static FLibrary& get() {
    if (_instance == nullptr) {
      _instance = std::unique_ptr<FLibrary>(new FLibraryDefault());
    }
    return *_instance;
  }

  bool supports(FuncSymbol func) const override;

  void synthesize(
      const Out &out, const Value &value, Netlist &net) override;
  void synthesize(
      FuncSymbol func, const Out &out, const In &in, Netlist &net) override;
  void synthesize(
      const Out &out, const In &in, const ControlList &control, Netlist &net) override;

private:
  FLibraryDefault() {}
  ~FLibraryDefault() override {}

  static void synth_add(const Out &out, const In &in, Netlist &net);
  static void synth_sub(const Out &out, const In &in, Netlist &net);
  static void synth_mux(const Out &out, const In &in, Netlist &net);

  static void synth_adder(const Out &out, const In &in, bool plus_one, Netlist &net);
  static void synth_adder(unsigned z, unsigned c_out,
    unsigned x, unsigned y, unsigned c_in, Netlist &net);

  static Signal invert_if_negative(const ControlEvent &event, Netlist &net);

  template<GateSymbol G>
  static void synth_unary_bitwise_op (const Out &out, const In &in, Netlist &net);

  template<GateSymbol G>
  static void synth_binary_bitwise_op(const Out &out, const In &in, Netlist &net);

  static std::unique_ptr<FLibrary> _instance;
};

template<GateSymbol G>
void FLibraryDefault::synth_unary_bitwise_op(const Out &out, const In &in, Netlist &net) {
  assert(in.size() == 1);

  const GateIdList &x = in[0];
  assert(out.size() == x.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    Signal xi = net.always(x[i]);
    net.set_gate(out[i], G, { xi });
  }
}

template<GateSymbol G>
void FLibraryDefault::synth_binary_bitwise_op(const Out &out, const In &in, Netlist &net) {
  assert(in.size() == 2);

  const GateIdList &x = in[0];
  const GateIdList &y = in[1];
  assert(x.size() == y.size() && out.size() == x.size());

  for (std::size_t i = 0; i < out.size(); i++) {
    Signal xi = net.always(x[i]);
    Signal yi = net.always(y[i]);
    net.set_gate(out[i], G, { xi, yi });
  }
}

} // namespace eda::rtl::library
