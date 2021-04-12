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

#include <cstddef>
#include <string>

#include "event.h"
#include "function.h"
#include "variable.h"

namespace utopia {

class VNode final {
public:
  // Creation of v-nodes is carried out through IR.
  friend class Net;

  enum Kind {
    SRC,
    FUN,
    MUX,
    REG,
  };

  Kind kind() const { return _kind; }
  const Variable &var() const { return _var; }
  const Event& event() const { return _event; }
  Function fun() const { return _fun; }
  std::size_t arity() const { return _inputs.size(); }
  const VNode* input(size_t i) const { return _inputs[i]; }

private:
  VNode(Kind kind, const Variable &var, const Event &event, Function fun,
      const std::vector<VNode *> inputs):
    _kind(kind), _var(var), _event(event), _fun(fun), _inputs(inputs) {}

  VNode(Kind kind, const Variable &var, const Event &event, Function fun):
    _kind(kind), _var(var), _event(event), _fun(fun), _inputs() {}

  const Kind _kind;
  const Variable _var;
  const Event _event;
  const Function _fun;
  const std::vector<VNode *> _inputs;
};

} // namespace utopia

