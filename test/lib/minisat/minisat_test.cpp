//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "minisat/core/Solver.h"

#include <iostream>

bool testMiniSat() {
  using Minisat::mkLit;
  using Minisat::lbool;
  using Minisat::l_True;

  Minisat::Solver solver;

  // Create variables.
  auto x = solver.newVar();
  auto y = solver.newVar();
  auto z = solver.newVar();

  // Create the clauses.
  solver.addClause( mkLit(x),  mkLit(y),  mkLit(z));
  solver.addClause(~mkLit(x),  mkLit(y),  mkLit(z));
  solver.addClause( mkLit(x), ~mkLit(y),  mkLit(z));
  solver.addClause( mkLit(x),  mkLit(y), ~mkLit(z));

  // Check for solution and retrieve model if found.
  auto verdict = solver.solve();

  if (verdict) {
    std::cout << "SAT\n"
              << "Model found:\n";
    std::cout << "x = " 
              << (solver.modelValue(x) == l_True) 
              << '\n';
    std::cout << "y = " 
              << (solver.modelValue(y) == l_True)
              << '\n';
    std::cout << "z = " 
              << (solver.modelValue(z) == l_True)
              << '\n';
  } else {
    std::cout << "UNSAT\n";
  }

  return verdict;
}

TEST(MiniSatTest, MiniSatSimpleTest) {
  EXPECT_TRUE(testMiniSat());
}

