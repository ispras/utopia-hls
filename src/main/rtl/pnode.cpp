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

#include "rtl/pnode.h"
#include "rtl/vnode.h"

namespace eda {
namespace rtl {

std::ostream& operator <<(std::ostream &out, const PNode &pnode) {
  out << "always @(" << pnode.event() << ") begin" << std::endl;

  if (pnode.gsize() > 0) {  
    out << "  if (";
    bool separator = false;
    for (auto i = pnode.gbegin(); i != pnode.gend(); i++) {
      out << (separator ? " && " : "") << **i;
      separator = true;
    }
    out << ") begin" << std::endl;
  }

  for (auto i = pnode.abegin(); i != pnode.aend(); i++) {
    out << "    " << **i << std::endl;
  }

  if (pnode.gsize() > 0) {
    out << "  end" << std::endl;
  }

  out << "end";
  return out;
}

}} // namespace eda::rtl

