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

#include "rtl/net.h"
#include "utils/utils.h"

using namespace eda::utils;

namespace eda {
namespace rtl {

void Net::create() {
  assert(!_created);

  for (auto &[name, usage]: _vnodes_temp) {
    VNode *phi = usage.first;
    std::vector<VNode *> &defines = usage.second;

    // Multiple definitions <=> phi-node required.
    assert((phi != nullptr && defines.size() >= 2) ||
           (phi == nullptr && defines.size() == 1));

    // No multiplexing required.
    if (defines.size() == 1) {
      _vnodes.push_back(defines.front());
      continue;
    }

    // Multiplexing required.
    std::vector<VNode *> mux_inputs(2 * defines.size());
    Variable mux_output = phi->var();

    switch (phi->var().kind()) {
    // if (g[1]) { w <= f[1](...) }    w[1] <= f[1](...)
    // ...                          => ...               + w <= mux{ g[i] -> w[i] }
    // if (g[n]) { w <= f[n](...) }    w[n] <= f[n](...)
    case Variable::WIRE:
      // Creates the { w[i] } nodes and compose the mux inputs: { g[i] -> w[i] }.
      for (std::size_t i = 0; i < defines.size(); i++) {
        VNode *old_vnode = defines[i];

        assert(old_vnode->_pnode != nullptr);
        assert(!old_vnode->_pnode->_guard.empty());

        // Create a { w[i] <= f[i](...) } node.
        VNode *new_vnode = old_vnode->duplicate(unique_name(old_vnode->name()));
        _vnodes.push_back(new_vnode);

        // Guards come first: mux(g[1], ..., g[n]; w[1], ..., w[n]).
        mux_inputs[i] = old_vnode->_pnode->_guard.back();
        mux_inputs[i + defines.size()] = new_vnode;
      }

      // Connect the wire w/ the multiplexor: w <= mux{ g[i] -> w[i] }.
      phi->replace_with(VNode::MUX, mux_output, Event::always(), FuncSymbol::NOP, mux_inputs);
      _vnodes.push_back(phi);

      break;

    // if (g[1]) { r <= w[1] }    w <= mux{ g[i] -> w[i] }
    // ...                     => 
    // if (g[n]) { r <= w[n] }    r <= w
    case Variable::REG:
      // Compose the mux inputs { g[i] -> w[i] }.
      for (std::size_t i = 0; i < defines.size(); i++) {
        VNode *vnode = defines[i];
        assert(vnode->_pnode != nullptr);

        // Guards come first: mux(g[1], ..., g[n]; w[1], ..., w[n]).
        mux_inputs[i] = vnode->_pnode->_guard.back();
        mux_inputs[i + defines.size()] = vnode->_inputs.front();
      }

      // Create a wire w.
      Variable wire(unique_name(mux_output.name()), Variable::WIRE, mux_output.type());

      // Create a multiplexor: w <= mux{ g[i] -> w[i] }.
      VNode *mux = new VNode(VNode::MUX, wire, Event::always(), FuncSymbol::NOP, mux_inputs);
      _vnodes.push_back(mux);

      // Connect the register w/ the multiplexor via the wire: r <= w.
      phi->replace_with(VNode::REG, mux_output, defines.front()->event(), FuncSymbol::NOP, { mux });
      _vnodes.push_back(phi);

      break;
    }
  }

  _vnodes_temp.clear();
  _created = true;
}

std::ostream& operator <<(std::ostream &out, const Net &net) {
  for (auto i = net.pbegin(); i != net.pend(); i++) {
    out << **i << std::endl;
  }

  for (auto i = net.vbegin(); i != net.vend(); i++) {
    out << **i << std::endl;
  }

  return out;
}
 
}} // namespace eda::rtl

