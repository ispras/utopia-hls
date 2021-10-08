//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <hls/debugger/debugger.h>

using namespace eda::hls::model;
using namespace z3;

namespace eda::hls::debugger {

  std::unique_ptr<Verifier> Verifier::instance = nullptr;

  bool Verifier::equivalent(Model &left, Model &right) const {

    std::vector<Graph *> leftGraphs = left.graphs;
    std::vector<Graph *> rightGraphs = right.graphs;

    if (leftGraphs.size() != rightGraphs.size())
      return false;
    // TODO: check graph vectors here


    context ctx;

    // TODO: add function declarations here

    solver s(ctx);

    // TODO: add formulae here

    switch (s.check()) {
      case sat:
        std::cout << "Models are equivalent" << std::endl;
        return true;
      case unsat:
        std::cout << "Models are not equivalent" << std::endl;
        return false;
      case unknown:
      default:
        std::cout << "Z3 solver says \"unknown\"" << std::endl;
        break;
    }
  }
} // namespace eda::hls::debugger