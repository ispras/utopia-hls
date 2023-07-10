//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <limits>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

//===----------------------------------------------------------------------===//
//
// Fidducia-Mattheyses algorithm for 2-partitioning was taken from
// "VLSI Physical Design: From HyperGraph Partitioning to Timing Closure"
// by Andrew B. Kahng, Jens Lienig, Igor L. Markov, Jin Hu.
//
//===----------------------------------------------------------------------===//

struct Step {
  int node;
  int gain;
  int minRest;
};

class FMAlgo {

  // const G *graph;
  const std::vector<size_t> &eptr;
  const std::vector<unsigned int> &eind;
  const std::vector<int> &weights;
  std::vector<std::vector<int>> adjacentList;
  int maxDegree = 0;

  double r;
  int passes;
  int lower, upper;
  int area[2]{};
  std::vector<bool> sides;
  // side - distrib.
  std::vector<std::vector<int>> distrib;


public:
  FMAlgo(const std::vector<size_t> &eptr, const std::vector<unsigned int> &eind,
         const std::vector<int> &weights, double r, int passes);

  void fm();

  inline const std::vector<bool> &getSides() const {
    return sides;
  }

  inline const std::vector<std::vector<int>> &getDistrib() const {
    return distrib;
  };

private:
  void balanceCriterion();

  void randPartition();

  void countDistribution();

  int countGain(int node) const;

  int maxGain(const std::vector<std::list<int>> &gain) const;

  void tempMove(int node, std::vector<Step> &order,
                std::vector<std::list<int>> &gain,
                std::vector<std::list<int>::iterator> &ptrBucket,
                std::vector<int> &ptrGain);

  void moveNode(int node);

  int countMinCriterion(int node) const;

  void gainUpdate(int node, std::vector<std::list<int>> &gain,
                  std::vector<std::list<int>::iterator> &ptrBucket,
                  std::vector<int> &ptrGain);

  void
  bestMoves(const std::vector<Step> &order, int &Gm, int &m) const;

  void confirmMoves(const std::vector<Step> &order, int m);

  void countAdjacentList();
};