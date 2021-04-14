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
#include <vector>

#include "pnode.h"
#include "vnode.h"

namespace eda {
namespace ir {

/**
 * \brief An intermediate representation combining p- and v-nets.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Net final {
public:
  typedef typename std::vector<VNode *>::const_iterator const_viterator;
  typedef typename std::vector<PNode *>::const_iterator const_piterator;

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
    return add_vnode(new VNode(VNode::SRC, var, Event::always(), {}));
  }

  /// Creates and adds an f-node (s = function).
  VNode* add_fun(const Variable &var, Function fun, const std::vector<VNode *> &inputs) {
    assert(!_created);
    return add_vnode(new VNode(VNode::FUN, var, Event::always(), fun, inputs));
  }

  /// Creates and adds an m-node (m = multiplexor).
  VNode* add_mux(const Variable &var, const std::vector<VNode *> &inputs) {
    assert(!_created);
    return add_vnode(new VNode(VNode::MUX, var, Event::always(), inputs));
  }

  /// Creates and adds an r-node (r = register).
  VNode* add_reg(const Variable &var, const Event &event, VNode *input) {
    assert(!_created);
    return add_vnode(new VNode(VNode::REG, var, event, { input }));
  }

  /// Creates and adds a combinational p-node.
  PNode* add_cmb(const std::vector<VNode *> &guard, const std::vector<VNode *> &action) {
    assert(!_created);
    return add_pnode(new PNode(Event::always(), guard, action));
  }

  /// Creates and adds a sequential p-node.
  PNode* add_seq(const Event &event, const std::vector<VNode *> &guard,
      const std::vector<VNode *> &action) {
    assert(!_created);
    return add_pnode(new PNode(event, guard, action));
  }

  /// Creates the v-net according to the p-net.
  void create();

private:
  VNode* add_vnode(VNode *vnode) {
    _vnodes_temp[vnode->var().name()].push_back(vnode);
    return vnode;
  }

  PNode* add_pnode(PNode *pnode) {
    _pnodes.push_back(pnode);
    return pnode;
  }

  std::vector<VNode *> _vnodes;
  std::vector<PNode *> _pnodes;
 
  std::unordered_map<std::string, std::vector<VNode *>> _vnodes_temp;

  bool _created;
};

}} // namespace utopia::ir

