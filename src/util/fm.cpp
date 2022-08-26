//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "fm.h"

FMAlgo::FMAlgo(const std::vector<size_t> &eptr, const std::vector<unsigned int> &eind,
               const std::vector<int> &weights, double r, int passes) : eptr(
        eptr), eind(eind), weights(weights), r(r), passes(passes) {
  balanceCriterion();
  randPartition();
}

void FMAlgo::fm() {
  countAdjacentList();
  countDistribution();
  for (const auto &list: adjacentList) {
    if (static_cast<int>(list.size()) > maxDegree) {
      maxDegree = static_cast<int>(list.size());
    }
  }

  int m;
  int Gm = std::numeric_limits<int>::max();
  int iteration = 0;

  while (Gm > 0 && iteration++ < passes) {

    std::vector<Step> order;
    order.reserve(weights.size());

    size_t fixed = 0;
    std::vector<std::list<int>> gain(maxDegree * 2 + 1);
    std::vector<std::list<int>::iterator> ptrBucket(weights.size());
    std::vector<int> ptrGain(weights.size());

    for (size_t v = 0; v < weights.size(); ++v) {
      int index = maxDegree - countGain(static_cast<int>(v));
      ptrGain[v] = index;
      gain[index].push_front(static_cast<int>(v));
      ptrBucket[v] = gain[index].begin();
    }
    while (fixed < weights.size()) {

      int nodeMax = maxGain(gain);

      if (nodeMax == -1) {
        break;
      }

      tempMove(nodeMax, order, gain, ptrBucket, ptrGain);
      gainUpdate(nodeMax, gain, ptrBucket, ptrGain);
      ++fixed;
    }
    bestMoves(order, Gm, m);
    if (Gm >= 0) {
      confirmMoves(order, m);
    }
  }
}

void FMAlgo::balanceCriterion() {
  int max = 0;
  int total = 0;

  for (const auto &weight: weights) {
    if (max < weight) {
      max = weight;
    }
    total += weight;
  }

  lower = static_cast<int>(std::ceil(r * total)) - max;
  upper = static_cast<int>(r * total) + max;
}

void FMAlgo::randPartition() {
  area[0] = 0;
  area[1] = 0;
  int mid = lower + (upper - lower) / 2;

  sides.resize(weights.size());
  for (size_t node = 0; node < weights.size(); ++node) {
    if (area[0] < mid) {
      area[0] += weights[node];
      sides[node] = false;
    } else {
      area[1] += weights[node];
      sides[node] = true;
    }
  }
}

void FMAlgo::countDistribution() {
  distrib = std::vector<std::vector<int>>(2, std::vector<int>(eptr.size()));

  for (size_t v = 0; v < weights.size(); ++v) {

    bool side = sides[v];

    for (int e: adjacentList[v]) {
      ++distrib[side][e];
    }
  }
}

int FMAlgo::countGain(int node) const {
  int fs = 0;
  int ts = 0;
  bool side = sides[node];

  for (const auto &e: adjacentList[node]) {
    ts += (distrib[!side][e] == 0);
    fs += (distrib[side][e] == 1);
  }
  return fs - ts;
}

int FMAlgo::maxGain(const std::vector<std::list<int>> &gain) const {
  int maxNode = -1;
  int maxRest = 0;

  for (const auto &bucket: gain) {
    if (!bucket.empty()) {
      for (int node: bucket) {

        int curRest = countMinCriterion(node);

        if (curRest >= maxRest) {
          maxNode = node;
          maxRest = curRest;
        }
      }

      if (maxNode >= 0) {
        return maxNode;
      }
    }
  }
  return maxNode;
}

void FMAlgo::tempMove(int node, std::vector<Step> &order,
                      std::vector<std::list<int>> &gain,
                      std::vector<std::list<int>::iterator> &ptrBucket,
                      std::vector<int> &ptrGain) {
  int gainIt = ptrGain[node];
  order.push_back({node, maxDegree - gainIt, countMinCriterion(node)});
  gain[gainIt].erase(ptrBucket[node]);
  ptrBucket[node] = gain[gainIt].end();
  ptrGain[node] = -1;
  moveNode(node);
}

void FMAlgo::moveNode(int node) {
  bool side = sides[node];

  area[side] -= weights[node];
  area[!side] += weights[node];
  sides[node] = !side;
  for (auto &edge: adjacentList[node]) {
    --distrib[side][edge];
    ++distrib[!side][edge];
  }
}

int FMAlgo::countMinCriterion(int node) const {
  int narea[]{area[0], area[1]};
  bool side = sides[node];

  narea[side] -= weights[node];
  narea[!side] += weights[node];
  return std::min(upper - narea[0], narea[0] - lower);
}

void FMAlgo::gainUpdate(int node, std::vector<std::list<int>> &gain,
                        std::vector<std::list<int>::iterator> &ptrBucket,
                        std::vector<int> &ptrGain) {
  for (int e: adjacentList[node]) {

    bool toBlock = sides[node];
    int tf[2] = {distrib[toBlock][e] - 1, distrib[!toBlock][e]};

    if (tf[0] <= 1 || tf[1] <= 1) {
      for (size_t i = eptr[e - 1]; i < eptr[e]; ++i) {

        int nextV = eind[i];

        if (ptrGain[nextV] != -1) {
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

          int oldGain = maxDegree - ptrGain[nextV];
          int newGain = oldGain + inc;

          gain[maxDegree - oldGain].erase(ptrBucket[nextV]);
          gain[maxDegree - newGain].push_front(nextV);
          ptrBucket[nextV] = gain[maxDegree - newGain].begin();
          ptrGain[nextV] = maxDegree - newGain;
        }
      }
    }
  }
}

void FMAlgo::bestMoves(const std::vector<Step> &order, int &Gm, int &m) const {
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

void FMAlgo::confirmMoves(const std::vector<Step> &order, int m) {
  for (int i = static_cast<int>(order.size()) - 1; i != m; --i) {
    moveNode(order[i].node);
  }
}

void FMAlgo::countAdjacentList() {
  adjacentList = std::vector<std::vector<int>>(weights.size());

  for (size_t e = 1; e < eptr.size(); ++e) {
    for (size_t i = eptr[e - 1]; i < eptr[e]; ++i) {
      adjacentList[eind[i]].push_back(static_cast<int>(e));
    }
  }
}

