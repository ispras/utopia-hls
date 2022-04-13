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
  auto A = solver.newVar();
  auto B = solver.newVar();
  auto C = solver.newVar();

  // Create the clauses.
  solver.addClause( mkLit(A),  mkLit(B),  mkLit(C));
  solver.addClause(~mkLit(A),  mkLit(B),  mkLit(C));
  solver.addClause( mkLit(A), ~mkLit(B),  mkLit(C));
  solver.addClause( mkLit(A),  mkLit(B), ~mkLit(C));

  // Check for solution and retrieve model if found.
  auto sat = solver.solve();

  if (sat) {
    std::cout << "SAT\n"
              << "Model found:\n";
    std::cout << "A := " 
              << (solver.modelValue(A) == l_True) 
              << '\n';
    std::cout << "B := " 
              << (solver.modelValue(B) == l_True)
              << '\n';
    std::cout << "C := " 
              << (solver.modelValue(C) == l_True)
              << '\n';
  } else {
    std::cout << "UNSAT\n";
  }

  return sat;
}

TEST(MiniSatTest, MiniSatSimpleTest) {
  EXPECT_TRUE(testMiniSat());
}

