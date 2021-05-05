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
#include <unordered_map>

#include "rtl/parser/builder.h"

using namespace eda::rtl;

namespace eda::rtl::parser {

std::unique_ptr<Builder> Builder::_instance = nullptr;

Type Builder::to_type(const std::string &type) const {
  auto kind = (type.at(0) == 's' ? Type::SINT : Type::UINT);
  auto width = static_cast<std::size_t>(std::stoi(type.substr(2)));
  return Type(kind, width);
}

std::vector<bool> Builder::to_value(const std::string &value) const {
  std::vector<bool> result(value.size() - 2);
  for (std::size_t i = 2; i < value.size(); i++) {
    result[i - 2] = (value.at(i) != '0');
  }
  return result;
}

Variable Builder::to_var(const std::string &value) const {
  Type type(Type::UINT, value.size() - 2);
  Variable var("$" + value, Variable::WIRE, Variable::INNER, type);
  return var;
}

std::unique_ptr<Net> Builder::create() {
  auto net = std::make_unique<Net>();

  // Collect all declarations.
  std::unordered_map<std::string, Variable> variables;
  std::unordered_map<std::string, unsigned> def_count;

  for (const auto &decl: _model.decls) {
    auto v = variables.find(decl.name);
    if (v == variables.end()) {
      Variable var(decl.name, decl.kind, decl.bind, to_type(decl.type));
      variables.insert({ decl.name, var });
      def_count.insert({ decl.name, 0 });
    }
  }

  // Count variable definitions.
  for (const auto &proc: _model.procs) {
    for (const auto &assign: proc.action) {
      auto d = def_count.find(assign.out);
      if (d != def_count.end()) {
        d->second++;
      }
    }
  }

  // Create phi-nodes when required.
  std::unordered_map<std::string, VNode*> use_nodes;

  for (const auto &decl: _model.decls) {
    if (decl.bind == Variable::INPUT) {
      auto v = variables.find(decl.name);
      use_nodes[decl.name] = net->add_src(v->second);
    }
  }

  for (const auto &proc: _model.procs) {
    for (const auto &assign: proc.action) {
      auto v = variables.find(assign.out);
      if (def_count[assign.out] > 1) {
        use_nodes[assign.out] = net->add_phi(v->second);
      } else if (v->second.kind() == Variable::WIRE) {
        use_nodes[assign.out] = net->add_fun(v->second, assign.func, {});
      } else {
        use_nodes[assign.out] = net->add_reg(v->second, nullptr);
      }
    }
  }
 
  // Construct p-nodes.
  std::unordered_map<std::string, VNode*> val_nodes;

  for (const auto &proc: _model.procs) {
    Event event(proc.event, !proc.signal.empty() ? use_nodes[proc.signal] : nullptr);

    VNode::List guard;
    VNode::List action;

    if (!proc.guard.empty()) {
      guard.push_back(use_nodes[proc.guard]);
    }
 
    for (const auto &assign: proc.action) {
      std::vector<VNode *> inputs;

      for (const auto &in: assign.in) {
        if (in.at(0) == '0') {
	  VNode *cnode = nullptr;
          auto i = val_nodes.find(in);

	  if (i != val_nodes.end()) {
            cnode = i->second;
          } else {
            cnode = net->add_val(to_var(in), to_value(in));
	    val_nodes[in] = cnode;
          }

	  inputs.push_back(cnode);
          break;
        } else {
          auto i = use_nodes.find(in);
          assert(i != use_nodes.end());
          inputs.push_back(i->second);
        }
      }

      VNode *vnode = nullptr;
      auto v = variables.find(assign.out);

      if (def_count[assign.out] == 1) {
        vnode = use_nodes[assign.out];
        net->update(vnode, inputs);
      } else if (v->second.kind() == Variable::WIRE) {
        vnode = net->add_fun(v->second, assign.func, inputs);
      } else {
	vnode = net->add_reg(v->second, inputs.front());
      }

      action.push_back(vnode);
    }

    net->add_seq(event, guard, action);
  }

  return net;
}

} // namespace eda::rtl::parser
