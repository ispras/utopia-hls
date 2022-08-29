//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class HyperGraph {
  std::vector<int> weights;
  std::vector<size_t> eptr;
  std::vector<unsigned int> eind;


public:
  explicit HyperGraph(std::ifstream &fin);

  HyperGraph(std::vector<int> *weights, std::vector<size_t> *eptr,
             std::vector<unsigned int> *eind) {
    /*std::swap(this->weights, *weights);
    std::swap(this->eptr, *eptr);
    std::swap(this->eind, *eind);*/
    this->weights = std::vector<int>(*weights);
    this->eptr = std::vector<size_t>(*eptr);
    this->eind = std::vector<unsigned int>(*eind);
  }

  HyperGraph(size_t nodesSize, int seed);

  inline const std::vector<int> &getWeights() const { return weights; }

  inline const std::vector<size_t> &getEptr() const { return eptr; }

  inline const std::vector<unsigned int> &getEind() const { return eind; }

  /**
   * Connects the hypergraph by |V| / (step - 1) edges the way that
   * last node of i(current) edge is the first node of the the i+1(next) edge
   * and each edge has step number of nodesIDs with consecutive numbers.
   * @param step
   */
  void addLinkedEdges(size_t step);

  /**
   * Sets rand hyperedges between nodesIDs.
   * @param edgeNumber number of edges in hypergraph.
   * @param edgeSize limit number of nodesIDs in one heperedge.
   */
  void setRndEdges(int edgeNumber, int edgeSize);

  /**
   * Assigns evenly distributed weights between 1 and upperLimit for all nodesIDs.
   */
  void setRndWeights(int upperLimit);

  /**
   * Counts cutset according to given distribution map.
   * @param distrib array for 2 partitions of map where key is edge number,
   * value is number of its nodesIDs in the partition.
   * @return cutset(number of edges that have its nodesIDs more than in 1 partition).
   */
  int countCutSet(const std::vector<std::vector<int>> &distrib) const;
 
  template<typename T>
  int countCutSet(const T &sides) const {
    int cutset = 0;
    for (size_t i = 0; i < eptr.size() - 1; ++i) {
      auto side = sides[eind[eptr[i]]];
      for (size_t j = eptr[i]; j < eptr[i + 1]; ++j) {
        int node = eind[j];
        if(sides[node]!=side) {
          ++cutset;
          break;
        }
      }
    }
    return cutset;
  }

  /**
   * Prints distribution of nodesIDs numbers among the parts to console.
   * For each partition prints total area according to node weights.
   * @param sides nodesIDs distribution among their parts.
   */
  void print(const std::vector<bool> &sides) const;

  void printArea(const std::vector<bool> &sides) const;

  /**
   * Outputs partitioned graph in dot format to the file with given path.
   * Uses method dotOutput for the task.
   * @param filename path the the file to ouput to.
   * @param sides nodesIDs distribution among their parts.
   * @return false if file can not be opened, true in case of success.
   */
  template<typename T>
  bool graphOutput(const std::string &filename,
                   const T &sides) const {
    std::ofstream fout(filename, std::ios_base::app);
    if (fout.is_open()) {
      dotOutput(fout, sides);
      fout.close();
      return true;
    }
    return false;
  }

  /**
 * Outputs partitioned graph in dot format to the file.
 * @param fout stream to output graph to.
 * @param sides nodesIDs distribution among their parts.
 */
  template<typename T>
  void dotOutput(std::ofstream &fout,
                 const T &sides) const {
    const char *colors[] = {"blue", "red"};
    fout << "graph partitioned {\n";
    for (int side = 0; side < 2; ++side) {
      fout << "\tsubgraph cluster_" << side << " {\n";
      for (size_t i = 0; i < weights.size(); ++i) {
        if (sides[i] == side) {
          fout << "\t\tnodes" << i;
          fout << ";\n";
        }
      }
      for (size_t i = 0; i < eptr.size() - 1; ++i) {
        bool there = false;
        for (size_t j = eptr[i]; j < eptr[i + 1]; ++j) {
          there |= (sides[eind[j]] != side);
        }
        if (!there) {
          fout << "\t\tedges" << i << "[shape=point];\n";
        }
      }
      fout << "\t\tcolor=" << colors[side] << ";\n";
      fout << "\t}\n";
    }
    for (size_t i = 0; i < eptr.size() - 1; ++i) {
      fout << "\tedges" << i << "[shape=point];\n";
      for (size_t j = eptr[i]; j < eptr[i + 1]; ++j) {
        int nodet = eind[j];
        if (nodet & 1) {
          fout << "\tedges" << i << " -- nodes" << nodet << ";\n";
        } else {
          fout << "\tnodes" << nodet << " -- edges" << i << ";\n";
        }
      }
    }
    fout << "}" << std::endl;
  }
};
