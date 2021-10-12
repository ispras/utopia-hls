//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include "z3++.h"

#include <list>
#include <memory>

using namespace eda::hls::model;
using namespace z3;

namespace eda::hls::debugger {

class Verifier final {

public:
  static Verifier& get() {
    if (instance == nullptr) {
      instance = std::unique_ptr<Verifier>(new Verifier());
    }
    return *instance;
  }

  bool equivalent(Model &left, Model &right) const;

private:
  Verifier() {}

  static std::unique_ptr<Verifier> instance;

  bool match(std::vector<Graph*> left,
      std::vector<Graph*> right,
      std::list<std::pair<Graph*, Graph*>> &matches) const;

  void to_expr(Graph *graph, context &ctx, std::vector<expr *> nodes) const;
  z3::func_decl mkFunction(const char *name, sort fSort) const;
};
} // namespace eda::hls::debugger
