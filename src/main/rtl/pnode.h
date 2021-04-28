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

#include <iostream>
#include <vector>

#include "rtl/event.h"
#include "rtl/vnode.h"

namespace eda::rtl {

/**
 * \brief Represents a p-node (p = process), a guarded action.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class PNode final {
  // Creation.
  friend class Net;

public:
  typedef std::vector<PNode *> List;

  const Event& event() const { return _event; }

  std::size_t gsize() const { return _guard.size(); }
  const VNode::List& guard() const { return _guard; }
  const VNode* guard(std::size_t i) const { return _guard[i]; }

  std::size_t asize() const { return _action.size(); }
  const VNode::List& action() const { return _action; }
  const VNode* action(std::size_t i) const { return _action[i]; }

private:
  PNode(const Event &event, const VNode::List &guard, const VNode::List &action):
      _event(event), _guard(guard), _action(action) {
    for (auto *vnode: guard) {
      vnode->set_pnode(this);
    }
    for (auto *vnode: action) {
      vnode->set_pnode(this);
    }
  }

  PNode(const VNode::List &guard, const VNode::List &action):
      PNode(Event(), guard, action) {}

  // The execution trigger (posedge, always, etc.).
  const Event _event;
  // The last v-node is the guard bit.
  VNode::List _guard;
  // The non-blocking assignments.
  VNode::List _action;
};

std::ostream& operator <<(std::ostream &out, const PNode &pnode);

} // namespace eda::rtl
