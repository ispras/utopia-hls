//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "HIL/Dialect.h"
#include "HIL/Ops.h"
#include "hls/model/model.h"

#include <z3++.h>

#include <list>
#include <memory>

using namespace mlir::hil;

namespace eda::hls::debugger {

class EqChecker final {

public:
  static EqChecker& get() {
    if (instance == nullptr) {
      instance = std::unique_ptr<EqChecker>(new EqChecker());
    }
    return *instance;
  }

  // Checks if models are equivalent.
  bool equivalent(mlir::hil::Model &left, mlir::hil::Model &right) const;

private:
  EqChecker() {}

  static std::unique_ptr<EqChecker> instance;

  /* Methods that implement equivalence checking sub-tasks. */

  /// Checks if collections contain nodes with same names.
  bool match(const std::vector<mlir::hil::Node> &left,
      const std::vector<mlir::hil::Node> &right,
      std::list<std::pair<mlir::hil::Node, mlir::hil::Node>> &matched) const;

  /* Methods for model-to-solver interaction. */

  /// Creates formal expressions for the specified graph.
  void createExprs(mlir::hil::Graph &graph, z3::context &ctx,
      z3::expr_vector &nodes) const;

  /// Creates constant expression for the channel's binding.
  z3::expr toConst(mlir::hil::Chan &ch,
      const mlir::hil::BindingAttr &bnd,
      z3::context &ctx) const;

  /// Creates constant expression for the node.
  z3::expr toConst(mlir::hil::Node &node, z3::context &ctx) const;

  /// Creates constant expression for the channel.
  z3::expr toConst(mlir::hil::Chan &ch, z3::context &ctx) const;

  /// Creates input function call for the node-chan pair.
  z3::expr toInFunc(mlir::hil::Node &node, mlir::hil::Chan &ch,
    z3::context &ctx) const;

  /// Calculates sort of the node.
  z3::sort getSort(mlir::hil::Node &node, z3::context &ctx) const;

  /// Calculates sort of the named port.
  z3::sort getSort(mlir::hil::PortAttr port, z3::context &ctx) const;

  /// Returns sorts of the node's inputs.
  z3::sort_vector getInSorts(mlir::hil::Node &node, z3::context &ctx) const;

  /// Returns arguments for function call that is constructed from the node.
  z3::expr_vector getFuncArgs(mlir::hil::Node &node, z3::context &ctx) const;

  /* Utility methods to operate with model. */

  /// Returns parent model's name for the node.
  std::string getModelName(mlir::hil::Node &node) const;

  /// Returns parent model's name for the channel.
  std::string getModelName(mlir::hil::Chan &ch) const;
};
} // namespace eda::hls::debugger
