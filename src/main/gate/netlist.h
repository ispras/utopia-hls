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
#include <vector>

#include "gate/gate.h"

namespace eda {
namespace rtl {

class Net;
class VNode;

}} // namespace eda::rtl

namespace eda {
namespace gate {

/**
 * \brief Represents a gate-level netlist.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Netlist final {
  // Debug print.
  friend std::ostream& operator <<(std::ostream &out, const Netlist &netlist);

public:
  typedef typename std::vector<Gate *>::const_iterator const_iterator;

  Netlist() {
    _gates.reserve(1024*1024);
    _gates_id.reserve(1024*1024);
  } 

  std::size_t size() const { return _gates.size(); }
  const_iterator begin() const { return _gates.cbegin(); }
  const_iterator end() const { return _gates.cend(); }

  /// Synthesizes the gate-level netlist from the RTL-level net.
  void create(const eda::rtl::Net &net);

private:
  unsigned gate_id(const eda::rtl::VNode *vnode);
  void allocate_gates(const eda::rtl::VNode *vnode);
  void handle_src(const eda::rtl::VNode *vnode);
  void handle_fun(const eda::rtl::VNode *vnode);
  void handle_mux(const eda::rtl::VNode *vnode);
  void handle_reg(const eda::rtl::VNode *vnode);

  std::vector<Gate *> _gates;

  // Maps vnodes to the identifiers of their lower bits' gates.
  std::unordered_map<std::string, unsigned> _gates_id;
};

std::ostream& operator <<(std::ostream &out, const Netlist &netlist);

}} // namespace eda::gate

