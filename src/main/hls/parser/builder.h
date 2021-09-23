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

#include <memory>
#include <string>
#include <vector>

#include "hls/model/model.h"

using namespace eda::hls::model;

namespace eda::hls::parser {

/**
 * \brief Helps to contruct the IR from source code.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Builder final {
public:
  static Builder& get() {
    if (_instance == nullptr) {
      _instance = std::unique_ptr<Builder>(new Builder());
    }
    return *_instance;
  }

  std::unique_ptr<Model> create();

  void start_model() {}
  void end_model() {}

  void start_nodetype() {}
  void end_nodetype(const std::string &name, const std::string &latency) {}

  void start_output_args() {}
  void add_arg(const std::string &type, const std::string &name, const std::string &flow) {}

  void start_graph() {}
  void end_graph(const std::string &name) {}

  void add_chan(const std::string &type, const std::string &name) {}

  void start_node() {}
  void end_node(const std::string &name) {}

  void start_output_param() {}
  void add_param(const std::string &name) {}

private:
  Builder() {}

  static std::unique_ptr<Builder> _instance;
};

} // namespace eda::hls::parser
