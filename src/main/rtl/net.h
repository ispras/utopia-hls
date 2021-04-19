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
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rtl/pnode.h"
#include "rtl/vnode.h"

namespace eda {
namespace rtl {

/**
 * \brief An intermediate representation combining p- and v-nets.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Net final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Net &net);

public:
  typedef std::vector<VNode *> VNodeList;
  typedef std::vector<PNode *> PNodeList;
  typedef std::vector<Event> EventList;

  typedef typename VNodeList::const_iterator const_viterator;
  typedef typename PNodeList::const_iterator const_piterator;

  Net(): _created(false) {
    _vnodes.reserve(1024*1024);
    _pnodes.reserve(1024*1024);
    _vnodes_temp.reserve(1024*1024);
  } 

  std::size_t vsize() const { return _vnodes.size(); }
  const_viterator vbegin() const { return _vnodes.cbegin(); }
  const_viterator vend() const { return _vnodes.cend(); }

  std::size_t psize() const { return _pnodes.size(); }
  const_piterator pbegin() const { return _pnodes.cbegin(); }
  const_piterator pend() const { return _pnodes.cend(); }

  /// Creates and adds an s-node (s = source).
  VNode* add_src(const Variable &var) {
    assert(!_created);
    return add_vnode(new VNode(VNode::SRC, var, {}, FuncSymbol::NOP, {}));
  }

  /// Creates and adds an f-node (s = function).
  VNode* add_fun(const Variable &var, FuncSymbol func, const VNodeList &inputs) {
    assert(!_created);
    return add_vnode(new VNode(VNode::FUN, var, {}, func, inputs));
  }

  /// Creates and adds a phi-node (unspecified multiplexor).
  VNode *add_phi(const Variable &var) {
    assert(!_created);
    return add_vnode(new VNode(VNode::MUX, var, {}, FuncSymbol::NOP,  {}));
  }

  /// Creates and adds an m-node (m = multiplexor).
  VNode* add_mux(const Variable &var, const VNodeList &inputs) {
    assert(!_created);
    return add_vnode(new VNode(VNode::MUX, var, {}, FuncSymbol::NOP, inputs));
  }

  /// Creates and adds an r-node (r = register).
  VNode* add_reg(const Variable &var, const EventList &events, VNode *input) {
    assert(!_created);
    return add_vnode(new VNode(VNode::REG, var, events, FuncSymbol::NOP, { input }));
  }

  /// Creates and adds a combinational p-node.
  PNode* add_cmb(const VNodeList &guard, const VNodeList &action) {
    assert(!_created);
    return add_pnode(new PNode(Event::always(), guard, action));
  }

  /// Creates and adds a sequential p-node.
  PNode* add_seq(const Event &event, const VNodeList &guard, const VNodeList &action) {
    assert(!_created);
    return add_pnode(new PNode(event, guard, action));
  }

  /// Creates the v-net according to the p-net.
  void create();

private:
  void mux_wire_defines(VNode *phi, const VNodeList &defines);

  void mux_reg_defines(VNode *phi, const VNodeList &defines);
  std::vector<std::pair<Event, VNodeList>> group_reg_defines(const VNodeList &defines);
  VNode* create_mux(const Variable &output, const VNodeList &defines);

  VNode* add_vnode(VNode *vnode) {
    auto &usage = _vnodes_temp[vnode->var().name()];
    if (vnode->kind() == VNode::MUX) {
       usage.first = vnode;
    } else {
       usage.second.push_back(vnode);
    }
    return vnode;
  }

  PNode* add_pnode(PNode *pnode) {
    _pnodes.push_back(pnode);
    return pnode;
  }

  VNodeList _vnodes;
  PNodeList _pnodes;
 
  /// Maps a variable x to the <phi(x), {def(x), ..., def(x)}> structure.
  std::unordered_map<std::string, std::pair<VNode *, VNodeList>> _vnodes_temp;

  bool _created;
};

std::ostream& operator <<(std::ostream &out, const Net &net);

}} // namespace eda::rtl

