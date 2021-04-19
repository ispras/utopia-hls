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

  for (auto &[_, usage]: _vnodes_temp) {
    assert(!_.empty());

    VNode *phi = usage.first;
    VNodeList &defines = usage.second;

    // Multiple definitions <=> phi-node required.
    assert((phi != nullptr && defines.size() >= 2) ||
           (phi == nullptr && defines.size() == 1));

    // No multiplexing required.
    if (defines.size() == 1) {
      // TODO: Set event if register.
      _vnodes.push_back(defines.front());
      continue;
    }

    // Multiplexing required.
    switch (phi->var().kind()) {
    case Variable::WIRE:
      mux_wire_defines(phi, defines);
      break;
    case Variable::REG:
      mux_reg_defines(phi, defines);
      break;
    }
  }

  _vnodes_temp.clear();
  _created = true;
}

// if (g[1]) { w <= f[1](...) }    w[1] <= f[1](...)
// ...                          => ...               + w <= mux{ g[i] -> w[i] }
// if (g[n]) { w <= f[n](...) }    w[n] <= f[n](...)
void Net::mux_wire_defines(VNode *phi, const VNodeList &defines) {
  const std::size_t n = defines.size();
  assert(n > 1);

  // Create the { w[i] } nodes and compose the mux inputs: { g[i] -> w[i] }.
  VNodeList inputs(2 * n);

  for (std::size_t i = 0; i < n; i++) {
    VNode *old_vnode = defines[i];

    assert(old_vnode->pnode() != nullptr);
    assert(!old_vnode->pnode()->_guard.empty());

    // Create a { w[i] <= f[i](...) } node.
    VNode *new_vnode = old_vnode->duplicate(unique_name(old_vnode->name()));
    _vnodes.push_back(new_vnode);

    // Guards come first: mux(g[1], ..., g[n]; w[1], ..., w[n]).
    inputs[i] = old_vnode->pnode()->_guard.back();
    inputs[i + n] = new_vnode;
  }

  // Connect the wire w/ the multiplexor: w <= mux{ g[i] -> w[i] }.
  Variable output = phi->var();
  phi->replace_with(VNode::MUX, output, {}, FuncSymbol::NOP, inputs);

  _vnodes.push_back(phi);
}

// @(event): if (g[1]) { r <= w[1] }    w <= mux{ g[i] -> w[i] }
// ...                     =>
// @(event): if (g[n]) { r <= w[n] }    @(event): r <= w
void Net::mux_reg_defines(VNode *phi, const VNodeList &defines) {
  std::vector<std::pair<Event, VNodeList>> groups = group_reg_defines(defines);

  Variable output = phi->var();

  EventList events;
  VNodeList inputs;

  for (const auto &[event, defines]: groups) {
    // Create a wire w for the given event.
    const std::string name = output.name() + "$" + event.node()->name();
    Variable wire(name, Variable::WIRE, output.type());

    // Create a multiplexor: w <= mux{ g[i] -> w[i] }.
    VNode *mux = create_mux(wire, defines);
    _vnodes.push_back(mux);

    events.push_back(event);
    inputs.push_back(mux);
  }

  // Connect the register w/ the multiplexor(s) via the wire(s): r <= w.
  phi->replace_with(VNode::REG, output, events, FuncSymbol::NOP, inputs);
  _vnodes.push_back(phi);
}

std::vector<std::pair<Event, Net::VNodeList>> Net::group_reg_defines(const VNodeList &defines) {
  const Event *clock = nullptr;
  const Event *level = nullptr;

  VNodeList clock_defines;
  VNodeList level_defines;

  // Collect all the events triggering the register.
  for (VNode *vnode: defines) {
    assert(vnode != nullptr && vnode->pnode() != nullptr);

    const Event &event = vnode->pnode()->event();
    assert(event.edge() || event.level());

    if (event.edge()) {
      // At most one edge-triggered event (clock) is allowed.
      assert(clock == nullptr || *clock == event);
      clock = &event;
      clock_defines.push_back(vnode);
    } else {
      // At most one level-triggered event (enable or reset) is allowed.
      assert(level == nullptr || *level == event);
      level = &event;
      level_defines.push_back(vnode);
    }
  }

  std::vector<std::pair<Event, VNodeList>> groups;
  if (clock != nullptr) {
    groups.push_back({ *clock, clock_defines });
  }
  if (level != nullptr) {
    groups.push_back({ *level, level_defines });
  }

  return groups;
}

VNode* Net::create_mux(const Variable &output, const VNodeList &defines) {
  const std::size_t n = defines.size();
  assert(n > 0);

  // Multiplexor is not required.
  if (n == 1) {
    VNode *vnode = defines.front();
    return new VNode(VNode::FUN, output, {}, FuncSymbol::NOP, { vnode->_inputs.front() }); 
  }

  // Compose the mux inputs { g[i] -> w[i] }.
  VNodeList inputs(2 * n);

  for (std::size_t i = 0; i < n; i++) {
    VNode *vnode = defines[i];
    assert(vnode->pnode() != nullptr);

    // Guards come first: mux(g[1], ..., g[n]; w[1], ..., w[n]).
    inputs[i] = vnode->pnode()->_guard.back();
    inputs[i + n] = vnode->_inputs.front();
  }

  // Create a multiplexor: w <= mux{ g[i] -> w[i] }.
  return new VNode(VNode::MUX, output, {}, FuncSymbol::NOP, inputs);
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

