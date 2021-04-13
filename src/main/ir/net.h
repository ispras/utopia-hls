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

#include <vector>

#include "pnode.h"
#include "vnode.h"

namespace utopia {

/**
 * \brief An intermediate representation combining p- and v-nets.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Net final {
public:
  typedef typename std::vector<VNode *>::const_iterator const_viterator;
  typedef typename std::vector<PNode *>::const_iterator const_piterator;

  Net(): _vnodes(1024*1024), _pnodes(1024*1024) {}

  std::size_t vsize() const { return _vnodes.size(); }
  const VNode* vnode(size_t i) const { return _vnodes[i]; }

  const_viterator vbegin() const { return _vnodes.cbegin(); }
  const_viterator vend() const { return _vnodes.cend(); }

  std::size_t psize() const { return _pnodes.size(); }
  const PNode* pnode(size_t i) const { return _pnodes[i]; }

  const_piterator pbegin() const { return _pnodes.cbegin(); }
  const_piterator pend() const { return _pnodes.cend(); }

  VNode* add_src(const Variable &var) {
    return add_vnode(new VNode(VNode::SRC, var, Event::always(), Function::NOP));
  }

  VNode* add_fun(const Variable &var, Function fun, const std::vector<VNode *> &inputs) {
    return add_vnode(new VNode(VNode::FUN, var, Event::always(), fun, inputs));
  }

  VNode* add_mux(const Variable &var, const std::vector<VNode *> &inputs) {
    return add_vnode(new VNode(VNode::MUX, var, Event::always(), Function::MUX, inputs));
  }

  VNode* add_reg(const Variable &var, const Event &event, const VNode *input) {
    std::vector<VNode *> inputs = { inputs };
    return add_vnode(new VNode(VNode::REG, var, event, Function::NOP, inputs));
  }

  PNode* add_cmb(const std::vector<VNode *> &action) {
    return add_pnode(new PNode(Event::always(), action));
  }

  PNode* add_cmb(const std::vector<VNode *> &guard, const std::vector<VNode *> &action) {
    return add_pnode(new PNode(Event::always(), guard, action));
  }

  PNode* add_seq(const Event &event, const std::vector<VNode *> &action) {
    return add_pnode(new PNode(event, action));
  }

  PNode* add_seq(const Event &event, const std::vector<VNode *> &guard,
      const std::vector<VNode *> &action) {
    return add_pnode(new PNode(event, guard, action));
  }

private:
  VNode* add_vnode(VNode *vnode) {
    _vnodes.push_back(vnode);
    return vnode;
  }

  PNode* add_pnode(PNode *pnode) {
    _pnodes.push_back(pnode);
    return pnode;
  }

  std::vector<VNode *> _vnodes;
  std::vector<PNode *> _pnodes;
};

} // namespace utopia

