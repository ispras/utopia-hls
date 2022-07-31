//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class HyperGraph {
  struct MultiEdge {
    std::vector<int> nodesIDs;
  };

  struct Node {
    int weight;
    std::vector<int> next;
  };

  std::vector<int> nodesNumber;
  std::vector<Node> nodes;
  std::vector<MultiEdge> edges;

public:
  explicit HyperGraph(size_t nodesSize);

  HyperGraph(size_t nodesSize, int seed);

  inline const std::vector<int>& getSources() const { return nodesNumber; }

  inline const std::vector<int>&
  getOutEdges(int node) const { return nodes[node].next; }

  inline const std::vector<int>&
  getEdgeNodes(int edge) const { return edges[edge].nodesIDs; }

  inline std::size_t
  nNodes(int edge) const { return edges[edge].nodesIDs.size(); }

  inline std::size_t
  nNodes() const { return nodes.size(); }

  inline int
  weight(int node) const { return nodes[node].weight; }

  /**
   * Adds edges to hypergraph according to input in fin.
   * The following format is expected:
   * N - Nodes number in hyperedge;
   * next N numbers are non-repeating numbers of nodesIDs
   * that hyperedge has to contain.
   */
  void addEdge(std::ifstream &fin);

  /**
   * Sets weights to all nodesIDs in hypergraph according to input in fin.
   * Input stream has to contain n numbers that correspond to weights.
   * @param fin
   */
  void setWeights(std::ifstream &fin);

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
  int countCutSet(const std::vector<std::unordered_map<int, int>> &distrib) const;

  /**
   * Prints distribution of nodesIDs numbers among the parts to console.
   * For each partition prints total area according to node weights.
   * @param sides nodesIDs distribution among their parts.
   */
  void print(const std::unordered_map<int, bool> *sides) const;

  /**
   * Outputs partitioned graph in dot format to the file with given path.
   * Uses method dotOutput for the task.
   * @param filename path the the file to ouput to.
   * @param sides nodesIDs distribution among their parts.
   * @return false if file can not be opened, true in case of success.
   */
  bool graphOutput(const std::string &filename,
                   const std::unordered_map<int, bool> *sides) const;

  /**
   * Outputs partitioned graph in dot format to the file.
   * @param fout stream to output graph to.
   * @param sides nodesIDs distribution among their parts.
   */
  void dotOutput(std::ofstream &fout,
                 const std::unordered_map<int, bool> *sides) const;
};

