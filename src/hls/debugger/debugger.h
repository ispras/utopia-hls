//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/model/model.h"

#include "z3++.h"

#include <list>
#include <memory>

using namespace eda::hls::model;

namespace eda::hls::debugger {

class EqChecker final {

public:
  static EqChecker& get() {
    if (instance == nullptr) {
      instance = std::unique_ptr<EqChecker>(new EqChecker());
    }
    return *instance;
  }

  /// Checks if models are equivalent.
  bool equivalent(const Model &left, const Model &right) const;

  /// Returns source nodes for the graph.
  std::vector<Node*> getSources(const Graph &graph) const;

  /// Returns sink nodes for the graph.
  std::vector<Node*> getSinks(const Graph &graph) const;

private:
  EqChecker() {}

  static std::unique_ptr<EqChecker> instance;

  /* Methods that implement equivalence checking siub-tasks. */

 /// Checks if collections contain nodes with same names.
  bool match(const std::vector<Node*> &left,
      const std::vector<Node*> &right,
      std::list<std::pair<Node*, Node*>> &matched) const;

  /* Methods for model-to-solver interaction. */

  /// Creates formulae (expressions) for the specified graph.
  void createExprs(const Graph &graph, z3::context &ctx,
      z3::expr_vector &nodes) const;

  /// Creates constant expression from the binding.
  z3::expr toConst(const Binding &bnd, z3::context &ctx) const;

  /// Creates constant expression from the node.
  z3::expr toConst(const Node &node, z3::context &ctx) const;

  /// Creates input function call for the node-chan pair.
  z3::expr toInFunc(const Node &node, const Chan &ch, z3::context &ctx) const;

  /// Calculates sort of the node.
  z3::sort getSort(const Node &node, z3::context &ctx) const;

  /// Calculates sort of the port.
  z3::sort getSort(const Port &port, z3::context &ctx) const;

  /// Returns sorts of the node's inputs.
  z3::sort_vector getInSorts(const Node &node, z3::context &ctx) const;

  /// Returns arguments for function call that is constructed from the node.
  z3::expr_vector getFuncArgs(const Node &node, z3::context &ctx) const;

  /* Utility methods to operate with model. */

  /// Returns parent model's name for the node.
  std::string getModelName(const Node &node) const;
};
} // namespace eda::hls::debugger
