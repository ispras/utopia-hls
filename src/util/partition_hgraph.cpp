//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "partition_hgraph.h"

HyperGraph::HyperGraph(std::ifstream &fin) {
  int nodeNumber, edgeNumber;
  std::string line;

  fin >> nodeNumber >> edgeNumber;
  std::getline(fin, line);
  weights.resize(nodeNumber);
  eptr.resize(edgeNumber + 1);

  eptr[0] = 0;
  for (int i = 0; i < edgeNumber; ++i) {
    int node;

    std::getline(fin, line);
    std::stringstream stream(line);
    while (stream >> node) {
      eind.push_back(node - 1);
    }
    eptr[i + 1] = (static_cast<int>(eind.size()));
  }

  for (int i = 0; i < nodeNumber; ++i) {
    fin >> weights[i];
  }
}

HyperGraph::HyperGraph(size_t nodesSize, int seed) {
  std::srand(seed);
  weights.resize(nodesSize);
}

void HyperGraph::setRndEdges(int edgeNumber, int edgeSize) {
  eptr.reserve(edgeNumber + 1);
  eptr.push_back(0);
  for (int i = 0; i < edgeNumber; ++i) {

    int size = rand() % (edgeSize - 2) + 2;

    for (int n = 0; n < size; ++n) {
      unsigned int node = rand() % weights.size();

      for (size_t j = eptr.back(); j < n + eptr.back(); ++j) {
        if (node == eind[j]) {
          --n;
          continue;
        }
      }
      eind.push_back(node);
    }
    eptr.push_back(eptr.back() + size);
  }
}

void HyperGraph::setRndWeights(int upperLimit) {
  for (auto &weight: weights) {
    weight = rand() % (upperLimit - 1) + 1;
  }
}

void HyperGraph::addLinkedEdges(size_t step) {
  eptr.push_back(0);
  for (size_t i = 0; i < weights.size(); i += (step - 1)) {
    for (size_t j = 0; j < step; ++j) {
      eind.push_back(static_cast<int>((j + i) % weights.size()));
    }
    eptr.push_back(static_cast<int>(step) + eptr.back());
  }
}

int
HyperGraph::countCutSet(const std::vector<std::vector<int>> &distrib) const {
  int cutset = 0;

  for (size_t i = 0; i < distrib[0].size(); ++i) {
    cutset += distrib[0][i] != 0 && distrib[1][i] != 0;
  }
  return cutset;
}

void HyperGraph::print(const std::vector<bool> &sides) const {
  for (int side = 0; side < 2; ++side) {
    std::cout << side << " : {";
    for (size_t i = 0; i < sides.size(); ++i) {
      if (sides[i] == side) {
        std::cout << " " << i;
      }
    }
    std::cout << " }\n";
  }
  std::cout << std::endl;
}

void HyperGraph::printArea(const std::vector<bool> &sides) const {
  int area[2]{};
  int number[2]{};

  for (size_t i = 0; i < sides.size(); ++i) {
    area[sides[i]] += weights[i];
    ++number[sides[i]];
  }
  std::cout << "Area[0] = " << area[0] << " number[0] = " << number[0] << '\n';
  std::cout << "Area[1] = " << area[1] << " number[1] = " << number[1] << std::endl;
}

