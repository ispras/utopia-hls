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

namespace eda::hls::debugger {

class Verifier final {

public:
  static Verifier& get() {
    if (instance == nullptr) {
      instance = std::unique_ptr<Verifier>(new Verifier());
    }
    return *instance;
  }

  bool equivalent(const Model &left, const Model &right) const;

private:
  Verifier() {}

  static std::unique_ptr<Verifier> instance;

  /* Methods taht implement equivalence checking steps. */

  bool match(const std::vector<Graph*> &left,
      const std::vector<Graph*> &right,
      std::pair<Graph*, Graph*> &matched) const;

  bool match(const std::vector<Node*> &left,
      const std::vector<Node*> &right,
      std::list<std::pair<Node*, Node*>> &matched) const;

  /* Methods for model-to-solver-format translation. */

  void createExprs(const Graph &graph, z3::context &ctx,
      z3::expr_vector &nodes) const;

  z3::expr toConst(const Binding &bnd, z3::context &ctx) const;
  z3::expr toConst(const Node &node, z3::context &ctx) const;
  z3::expr toInFunc(const Node &node, const Chan &ch, z3::context &ctx) const;
  z3::sort getSort(const Node &node, z3::context &ctx) const;
  z3::sort getSort(const Port &port, z3::context &ctx) const;
  z3::sort_vector getInSorts(const Node &node, z3::context &ctx) const;
  z3::expr_vector getFuncArgs(const Node &node, z3::context &ctx) const;

  /* Utility methods to operate with model. */

  std::string getModelName(const Node &node) const;
  std::vector<Node*> getSources(const Graph &graph) const;
  std::vector<Node*> getSinks(const Graph &graph) const;
};
} // namespace eda::hls::debugger
