//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "base/engine/context.h"

#include <string>

namespace eda::base::engine {

enum Status { SUCCESS, ERROR };

/**
 * \brief Utopia EDA engine interface.
 * \author <a href="mailto:kamkin@ispras.ru">Alexander Kamkin</a>
 */
class Engine {
public:
  Engine(const std::string &name): _name(name) {}
  virtual ~Engine() = default;

  Engine(const Engine&) = delete;
  Engine &operator=(const Engine&) = delete;

  /// Returns the engine name.
  std::string name() const { return _name; }

  /// Initializes the engine.
  virtual Status initialize() const { return SUCCESS; }
  /// Finalizes the engine.
  virtual Status finalize() const { return SUCCESS; }

  /// Executes the engine.
  virtual Status execute(const Context &context) const = 0;

private:
  const std::string _name;
};

} // namespace eda::base::engine
