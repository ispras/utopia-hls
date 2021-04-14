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

#include "net.h"
#include "utils.h"

namespace eda {
namespace ir {

void Net::create() {
  assert(!_created);

  for (auto &[name, vnodes]: _vnodes_temp) {
    assert(!vnodes.empty());

    VNode *vnode = vnodes.front();

    if (vnodes.size() == 1) {
      _vnodes.push_back(vnode);
      continue;
    }

    const Variable var = vnode->var();

    if (var.kind() == Variable::WIRE) {
      // { w <= f[i](...) } => { w[i] <= f[i](...) }, w <= mux{ w[i] }.
      std::vector<VNode *> new_vnodes(vnodes.size());
      std::vector<VNode *> mux_inputs(2 * vnodes.size());


      // Compose { c[i] } and { w[i] }.
      for (std::size_t i = 0; i < vnodes.size(); i++) {
        VNode *old_vnode = vnodes[i];
        VNode *new_vnode = old_vnode->duplicate(utils::format("%s$%d", old_vnode->name(), i));

        assert(old_vnode->_pnode != nullptr);
        mux_inputs[i] = old_vnode->_pnode->_guard.back();
        new_vnodes[i] = new_vnode;
      }

      // Add newly created { w[i] } to the v-net.
      _vnodes.insert(std::end(_vnodes), std::begin(new_vnodes), std::end(new_vnodes));

      // Compose { c[i], w[i] }.
      mux_inputs.insert(std::begin(mux_inputs) + vnodes.size(), std::begin(new_vnodes), std::end(new_vnodes));

      // Create a multiplexor.
      VNode *mux = new VNode(VNode::MUX, var, Event::always(), mux_inputs);
      _vnodes.push_back(mux);

      continue;
    }

    assert(var.kind() == Variable::REG);

    // { c[i]: r <= w[i] } => r <= w, w <= mux{ c[i] -> w[i] }.
    std::vector<VNode *> mux_inputs(2 * vnodes.size());
    std::size_t i = 0;

    // Compose { c[i] }.
    for (VNode *vnode: vnodes) {
      assert(vnode->_pnode != nullptr);
      mux_inputs[i++] = vnode->_pnode->_guard.back();
    }

    // Compose { c[i], w[i] }.
    for (VNode *vnode: vnodes) {
      mux_inputs[i++] = vnode->_inputs.front();
    }

    // Create a wire: w.
    Variable wire(utils::format("%s$wire", var.name()), Variable::WIRE, var.type());

    // Create a multiplexor: w <= mux{ c[i] -> w[i] }.
    VNode *mux = new VNode(VNode::MUX, wire, Event::always(), mux_inputs);
    _vnodes.push_back(mux);

    // Connect the register w/ the multiplexor via the wire: r <= w.
    VNode *reg = new VNode(VNode::REG, var, vnode->event(), { mux });
    _vnodes.push_back(reg);
  }

  _created = true;
}
 
}} // namespace eda::ir

