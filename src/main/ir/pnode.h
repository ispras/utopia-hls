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

#include "event.h"
#include "vnode.h"

namespace eda {
namespace ir {

/**
 * \brief Represents a p-node (p = process), a guarded action.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class PNode final {
  // Creation.
  friend class Net;
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const PNode &pnode);

public:
  typedef typename std::vector<VNode *>::const_iterator const_viterator;

  const Event& event() const { return _event; }

  std::size_t gsize() const { return _guard.size(); }
  const_viterator gbegin() const { return _guard.cbegin(); }
  const_viterator gend() const { return _guard.cend(); }

  std::size_t asize() const { return _action.size(); }
  const_viterator abegin() const { return _action.cbegin(); }
  const_viterator aend() const { return _action.cend(); }

private:
  PNode(const Event &event, const std::vector<VNode *> &guard, const std::vector<VNode *> &action):
      _event(event), _guard(guard), _action(action) {
    for (auto *vnode: guard) {
      vnode->set_pnode(this);
    }
    for (auto *vnode: action) {
      vnode->set_pnode(this);
    }
  }

  // The execution trigger (posedge, always, etc.).
  const Event _event;
  // The last v-node is the guard bit.
  std::vector<VNode *> _guard;
  // The non-blocking assignments.
  std::vector<VNode *> _action;
};

std::ostream& operator <<(std::ostream &out, const PNode &pnode);

}} // namespace eda::ir

