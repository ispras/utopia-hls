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

#include <iostream>

#include "rtl/vnode.h"

namespace eda {
namespace rtl {

static std::ostream& operator <<(std::ostream &out, const std::vector<bool> &value) {
  for (bool bit: value) {
    out << bit;
  }

  return out;
}

static std::ostream& operator <<(std::ostream &out, const VNode::List &vnodes) {
  bool separator = false;
  for (VNode *vnode: vnodes) {
    out << (separator ? ", " : "") << vnode->name();
    separator = true;
  }

  return out;
}

std::ostream& operator <<(std::ostream &out, const VNode &vnode) {
  switch (vnode.kind()) {
  case VNode::SRC:
    return out << "S{" << vnode.var() << "}";
  case VNode::VAL:
    return out << "C{" << vnode.var() << " <= " << vnode.value()<< "}";
  case VNode::FUN:
    return out << "F{" << vnode.var() << " <= " << vnode.func() << "(" << vnode.inputs() << ")}";
  case VNode::MUX:
    return out << "M{" << vnode.var() << " <= mux(" << vnode.inputs() << ")}";
  case VNode::REG:
    out << "R{";
    bool separator = false;
    for (std::size_t i = 0; i < vnode.esize(); i++) {
      out << (separator ? ", " : "") << vnode.event(i) << ": ";
      out << vnode.var() << " <= " << vnode.input(i)->name();
      separator = true;
    }
    return out << "}";
  }

  return out;
}

}} // namespace eda::rtl

