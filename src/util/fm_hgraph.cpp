//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "fm_hgraph.h"

HyperGraph::HyperGraph(size_t nodesSize) {
  nodes.resize(nodesSize);
  nodesNumber.resize(nodesSize);
  for (size_t i = 0; i < nodesSize; ++i) {
    nodesNumber[i] = static_cast<int>(i);
  }
}

HyperGraph::HyperGraph(size_t nodesSize, int seed) : HyperGraph(nodesSize) {
  std::srand(seed);
}

void HyperGraph::setWeights(std::ifstream &fin) {
  for (auto &node: nodes) {
    fin >> node.weight;
  }
}

void HyperGraph::addEdge(std::ifstream &fin) {
  int nodesNumber;
  fin >> nodesNumber;
  edges.emplace_back(MultiEdge{std::vector<int>(nodesNumber)});
  for (int &node: edges.back().nodesIDs) {
    fin >> node;
    nodes[node].next.push_back(static_cast<int>(edges.size()) - 1);
  }
}

void HyperGraph::setRndEdges(int edgeNumber, int edgeSize) {
  edges.resize(edgeNumber);
  std::vector<int> limits(edgeNumber);

  for (size_t i = 0; i < nodes.size(); ++i) {
    Node &nodet = nodes[i];
    int edge = rand() % edgeNumber;
    if (limits[edge] >= edgeSize) {
      edge = 0;
    }
    nodet.next.push_back(edge);
    edges[edge].nodesIDs.push_back(static_cast<int>(i));
    ++limits[edge];
  }
  for (size_t i = 0; i < limits.size(); ++i) {
    if (limits[i] == 0) {
      int node = rand() % (nodes.size() - 1);
      edges[i].nodesIDs.push_back(node);
      edges[i].nodesIDs.push_back(node + 1);
      nodes[node].next.push_back(i);
      nodes[node + 1].next.push_back(i);
    } else if (limits[i] == 1) {
      int node = (edges[i].nodesIDs[0] + 1) % static_cast<int>(nodes.size());
      edges[i].nodesIDs.push_back(node);
      nodes[node].next.push_back(i);
    }
  }
}

void HyperGraph::setRndWeights(int upperLimit) {
  for (auto &node: nodes) {
    node.weight = rand() % (upperLimit - 1) + 1;
  }
}

void HyperGraph::addLinkedEdges(size_t step) {
  edges = std::vector<MultiEdge>(
          (nodes.size() % (step - 1) != 0) + nodes.size() / (step - 1),
          MultiEdge{std::vector<int>(step)});
  int edge = 0;

  for (size_t i = 0; i < nodes.size(); i += (step - 1)) {
    for (size_t j = 0; j < step; ++j) {
      edges[edge].nodesIDs[j] = static_cast<int>((j + i) % nodes.size());
    }
    for (int node: edges[edge].nodesIDs) {
      nodes[node].next.push_back(edge);
    }
    ++edge;
  }
}

int HyperGraph::countCutSet(const std::vector<std::unordered_map<int, int>> &distrib) const {
  int cutset = 0;
  for (size_t i = 0; i < edges.size(); ++i) {
    auto itSide1 = distrib[0].find(static_cast<int>(i));
    auto itSide2 = distrib[1].find(static_cast<int>(i));
    cutset += itSide1 != distrib[0].end() && itSide1->second != 0 &&
              itSide2 != distrib[1].end() && itSide2->second != 0;
  }
  return cutset;
}

void HyperGraph::print(const std::unordered_map<int, bool> *sides) const {
  int area[2]{};
  for (int side = 0; side < 2; ++side) {
    std::cout << side << " : {";
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (sides->at(static_cast<int>(i)) == side) {
        area[side] += nodes[i].weight;
        std::cout << " " << i;
      }
    }
    std::cout << " } ";
    std::cout << area[side] << '\n';
  }
  std::cout << std::endl;
}

bool HyperGraph::graphOutput(const std::string &filename,
                             const std::unordered_map<int, bool> *sides) const {
  std::ofstream fout(filename);
  if (fout.is_open()) {
    dotOutput(fout, sides);
    fout.close();
    return true;
  }
  return false;
}

void HyperGraph::dotOutput(std::ofstream &fout,
                           const std::unordered_map<int, bool> *sides) const {
  const char *colors[] = {"blue", "red"};
  fout << "graph partitioned {\n";
  for (int side = 0; side < 2; ++side) {
    fout << "\tsubgraph cluster_" << side << " {\n";
    for (size_t i = 0; i < nodes.size(); ++i) {
      if (sides->at(i) == side) {
        fout << "\t\tnodes" << i;
        fout << ";\n";
      }
    }
    for (size_t i = 0; i < edges.size(); ++i) {
      bool there = false;
      for (int nodet: edges[i].nodesIDs) {
        there |= (sides->at(nodet) != side);
      }
      if (!there) {
        fout << "\t\tedges" << i << "[shape=point];\n";
      }
    }
    fout << "\t\tcolor=" << colors[side] << ";\n";
    fout << "\t}\n";
  }
  for (size_t i = 0; i < edges.size(); ++i) {
    fout << "\tedges" << i << "[shape=point];\n";
    for (int nodet: edges[i].nodesIDs) {
      if (nodet & 1) {
        fout << "\tedges" << i << " -- nodes" << nodet << ";\n";
      } else {
        fout << "\tnodes" << nodet << " -- edges" << i << ";\n";
      }
    }
  }
  fout << "}";
}

