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

#include <optional>

using namespace eda::rtl;

namespace eda::rtl::parser {

/**
 * \brief Helps to contruct the IR from source code.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Builder final {
public:
  static Type make_type(const std::string &token);
  static std::vector<bool> make_value(const std::string &token);

  void declare(const std::string& name, Variable::Kind kind, Variable::Bind bind, const Type &type) {
    Variable var(name, kind, bind, type);
    _variables[name] = _var;
  }

  std::optional<Variable> find(const std::string &name) {
    const auto i = _variables.find(name);
    return i != _variables.end() ? *i : {};
  }

private:
  std::unordered_map<std::string, Variable> _variables;
};

} // namespace eda::rtl::parser
