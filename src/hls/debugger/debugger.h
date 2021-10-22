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

  bool match(std::vector<Node*> left,
      std::vector<Node*> right,
      std::list<std::pair<Node*, Node*>> &matches) const;

  void to_expr(Graph *graph, context &ctx, expr_vector nodes) const;
  expr toConst(const Binding &bnd, context &ctx) const;
  expr toConst(const Node *node, context &ctx) const;
  expr toFunc(const Node *node, const Chan *ch, context &ctx) const;
  sort getSort(const Node *node, context &ctx) const;
  sort_vector getInSorts(const Node *node, context &ctx) const;
  expr_vector getArgs(const Node *node, context &ctx) const;
  std::vector<Node*> getSources(Graph *graph) const;
  std::vector<Node*> getSinks(Graph *graph) const;
};
} // namespace eda::hls::debugger
