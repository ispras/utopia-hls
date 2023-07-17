//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "HIL/Dialect.h"
#include "HIL/Ops.h"
#include "hls/model/model.h"

#include <z3++.h>

#include <list>
#include <memory>
#include <optional>

namespace eda::hls::eqchecker {

using Binding = const mlir::hil::BindingAttr;
using Chan = mlir::hil::Chan;
using ChanVector = std::vector<Chan>;
using CheckResult = z3::check_result;
using Context = z3::context;
using Expr = z3::expr;
using ExprVector = z3::expr_vector;
using FuncDecl = z3::func_decl;
using Graph = mlir::hil::Graph;
using Model = mlir::hil::Model;
using Node = mlir::hil::Node;
using NodePairList = std::list<std::pair<Node, Node>>;
using NodeVector = std::vector<Node>;
using OptionalGraph = std::optional<Graph>;
using Port = mlir::hil::PortAttr;
using Solver = z3::solver;
using Sort = z3::sort;
using SortVector = z3::sort_vector;

/**
 * \brief Equivalence checker for high-level models.
 *
 * The checker is based on uninterpreted functions formalism,
 * and uses Z3 solver to check program models.
 *
 * \author <a href="mailto:smolov@ispras.ru">Sergey Smolov</a>
 */
class EqChecker final {

public:
  static EqChecker& get() {
    if (instance == nullptr) {
      instance = std::unique_ptr<EqChecker>(new EqChecker());
    }
    return *instance;
  }

  /// Checks if models are equivalent.
  bool equivalent(Model &lhs, Model &rhs) const;

private:
  EqChecker() {}

  static std::unique_ptr<EqChecker> instance;

  /* Methods that implement equivalence checking sub-tasks. */

  /// Checks if collections contain nodes with same names.
  bool match(const NodeVector &left, const NodeVector &right,
      NodePairList &matched) const;

  /* Methods for model-to-solver interaction. */

  /// Extracts formal expressions from the specified graph.
  void makeExprs(Graph &graph, Context &ctx, ExprVector &nodes) const;

  /// Creates constant expression for the channel's binding.
  Expr toConst(Chan &ch, const Binding &bnd, Context &ctx) const;

  /// Creates constant expression for the node.
  Expr toConst(Node &node, Context &ctx) const;

  /// Creates constant expression for the channel.
  Expr toConst(Chan &ch, Context &ctx) const;

  /// Creates input function call for the node-chan pair.
  Expr toInFunc(Node &node, Chan &ch, Context &ctx) const;

  /// Calculates sort of the node.
  Sort getSort(Node &node, Context &ctx) const;

  /// Calculates sort of the named port.
  Sort getSort(Port port, Context &ctx) const;

  /// Returns sorts of the node's inputs.
  SortVector getInSorts(Node &node, Context &ctx) const;

  /// Returns arguments for function call that is constructed from the node.
  ExprVector getFuncArgs(Node &node, Context &ctx) const;
};
} // namespace eda::hls::eqchecker
