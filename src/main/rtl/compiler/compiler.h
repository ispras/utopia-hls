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
#include <unordered_map>

#include "gate/model/netlist.h"
#include "rtl/library/flibrary.h"
#include "rtl/model/net.h"
#include "rtl/model/vnode.h"

using namespace eda::gate::model;
using namespace eda::rtl::library;
using namespace eda::rtl::model;

namespace eda::rtl::compiler {

/**
 * \brief Implements a gate-level netlist compiler (logic synthesizer).
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Compiler final {
public:
  Compiler(Netlist &netlist, FLibrary &library):
    _netlist(netlist), _library(library) {
    _gates_id.reserve(1024*1024);
  }

  /// Compiles the gate-level netlist from the RTL net.
  void compile(const Net &net);

private:
  Netlist &_netlist;
  FLibrary &_library;

  unsigned gate_id(const VNode *vnode);
  void alloc_gates(const VNode *vnode);

  void synth_src(const VNode *vnode);
  void synth_val(const VNode *vnode);
  void synth_fun(const VNode *vnode);
  void synth_mux(const VNode *vnode);
  void synth_reg(const VNode *vnode);

  Netlist::Out out(const VNode *vnode);
  Netlist::In in(const VNode *vnode);

  // Maps vnodes to the identifiers of their lower bits' gates.
  std::unordered_map<std::string, unsigned> _gates_id;
};

} // namespace eda::rtl::compiler
