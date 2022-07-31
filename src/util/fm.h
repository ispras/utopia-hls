//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <limits>
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
// It is assumed that graph G contains the following methods:
//
// std::size_t  nNodes()           const;
// std::size_t  nEdges()           const;
// container<V> getSources()       const;
// container<E> getOutEdges(V nodes)   const;
// container<V> getEdgeNodes(E edges)  const;
// std::size_t  nNodes(E edges)        const;
// int          weight(V nodes)        const.
//===----------------------------------------------------------------------===//

template<typename V>
struct Step {
  V node;
  int gain, minRest;
};

template<typename G, typename V, typename E>
class FMAlgo {
  using GainNodeMap = std::multimap<int, V, std::greater<>>;
  using GainNodeIt = typename std::multimap<int, V, std::greater<>>::iterator;

  const G *graph;
  double r;
  int passes;
  int lower{}, upper{};
  int area[2]{};
  std::unordered_map<V, bool> sides;
    std::vector<std::unordered_map<E, int>> distrib
          = std::vector<std::unordered_map<E, int>>(2);


public:
  FMAlgo(const G *g, double r, int passes) : graph(g), r(r), passes(passes) {
    balanceCriterion();
    randPartition();
  }

  void fm() {
    countDistribution();

    int m;
    int Gm = std::numeric_limits<int>::max();
    int iteration = 0;

    while (Gm > 0 && iteration++ < passes) {

      std::vector<Step<V>> order;
      order.reserve(graph->nNodes());

      size_t fixed = 0;
      GainNodeMap gain;
      std::unordered_map<V, GainNodeIt> ptrGain(graph->nNodes());

      for (auto v: graph->getSources()) {
        ptrGain[v] = gain.emplace(countGain(v), v);
      }
      while (fixed < graph->nNodes()) {

        auto itMax = maxGain(gain);

        if (itMax == gain.end()) {
          break;
        }

        V node = itMax->second;

        tempMove(node, order, gain, ptrGain);
        gainUpdate(node, gain, ptrGain);
        ++fixed;
      }
      bestMoves(order, Gm, m);
      if (Gm >= 0) {
        confirmMoves(order, m);
      }
    }
  }

  const std::unordered_map<V, bool> *getSides() const {
    return &sides;
  }

  const std::vector<std::unordered_map<E, int>> &getDistrib() const {
    return distrib;
  };

private:
  void balanceCriterion() {
    int max = 0;
    int total = 0;

    for (const auto &node: graph->getSources()) {
      if (max < graph->weight(node)) {
        max = graph->weight(node);
      }
      total += graph->weight(node);
    }

    lower = static_cast<int>(std::ceil(r * total)) - max;
    upper = static_cast<int>(r * total) + max;
  }

  void randPartition() {
    area[0] = 0;
    area[1] = 0;
    int mid = lower + (upper - lower) / 2;

    for (const auto &node: graph->getSources()) {
      if (area[0] < mid) {
        area[0] += graph->weight(node);
        sides[node] = false;
      } else {
        area[1] += graph->weight(node);
        sides[node] = true;
      }
    }

  }

  void countDistribution() {
    for (const auto &v: graph->getSources()) {

      bool side = sides[v];

      for (const auto &e: graph->getOutEdges(v)) {
        ++distrib[side][e];
      }
    }
  }

  int countGain(const V &node) const {
    int fs = 0;
    int ts = 0;
    bool side = sides.at(node);

    for (const auto &e: graph->getOutEdges(node)) {
      ts += (distrib[side].at(e) == static_cast<int>(graph->nNodes(e)));
      fs += (distrib[side].at(e) == 1);
    }
    return fs - ts;
  }

  GainNodeIt maxGain(GainNodeMap &gain) {
    int curGain = gain.begin()->first;
    GainNodeIt maxnode = gain.end();
    int maxRest = 0;

    for (GainNodeIt it = gain.begin(); it != gain.end(); ++it) {

      auto gainVal = it->first;
      auto node = it->second;

      if (gainVal < curGain && maxnode != gain.end()) {
        return maxnode;
      }

      int curRest = countMinCriterion(node);

      if (curRest >= maxRest) {
        curGain = gainVal;
        maxnode = it;
        maxRest = curRest;
      }
    }
    return maxnode;
  }

  void tempMove(V v, std::vector<Step<V>> &order, GainNodeMap &gain,
                std::unordered_map<V, GainNodeIt> &ptrGain) {
    order.push_back({v, ptrGain[v]->first, countMinCriterion(v)});
    gain.erase(ptrGain[v]);
    ptrGain[v] = gain.end();
    movenode(v);
  }

  void movenode(V v) {
    bool side = sides[v];

    area[side] -= graph->weight(v);
    area[!side] += graph->weight(v);
    sides[v] = !side;
    for (auto &edge: graph->getOutEdges(v)) {
      --distrib[side][edge];
      ++distrib[!side][edge];
    }
  }

  int countMinCriterion(const V &v) const {
    int narea[]{area[0], area[1]};
    bool side = sides.at(v);

    narea[side] -= graph->weight(v);
    narea[!side] += graph->weight(v);
    return std::min(upper - narea[0], narea[0] - lower);
  }

  void gainUpdate(const V &v, GainNodeMap &gain,
                  std::unordered_map<V, GainNodeIt> &ptrGain) {
    for (const auto &e: graph->getOutEdges(v)) {

      bool toBlock = sides[v];
      int tf[2] = {distrib[toBlock][e] - 1, distrib[!toBlock][e]};

      if (tf[0] <= 1 || tf[1] <= 1) {
        for (const auto &nextV: graph->getEdgeNodes(e)) {
          if (ptrGain[nextV] != gain.end()) {
            int inc = 0;
            if (tf[0] == 0) {
              ++inc;
            } else if (tf[0] == 1 && sides[nextV] == toBlock) {
              --inc;
            }
            if (tf[1] == 0) {
              --inc;
            } else if (tf[1] == 1 && sides[nextV] != toBlock) {
              ++inc;
            }
            int newGain = ptrGain[nextV]->first + inc;
            gain.erase(ptrGain[nextV]);
            ptrGain[nextV] = gain.emplace(newGain, nextV);
          }
        }
      }
    }
  }

  void
  bestMoves(const std::vector<Step<V>> &order, int &Gm, int &m) const {
    Gm = std::numeric_limits<int>::min();
    m = 0;
    int sum = 0;

    for (size_t i = 0; i < order.size(); ++i) {
      sum += order[i].gain;
      if (sum > Gm ||
          (sum == Gm && order[i].minRest > order[m].minRest)) {
        Gm = sum;
        m = static_cast<int>(i);
      }
    }
  }

  void confirmMoves(const std::vector<Step<V>> &order, int m) {
    for (int i = static_cast<int>(order.size()) - 1; i != m; --i) {
      movenode(order[i].node);
    }
  }
};

