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

class Graph {
    struct MultiEdge {
        std::vector<int> vertexes;
    };

    struct Vertex {
        int weight;
        std::vector<int> next;
    };

    std::vector<int> vertexesNumber;
    std::vector<Vertex> v;
    std::vector<MultiEdge> e;

public:
    explicit Graph(size_t nodesSize);

    Graph(size_t nodesSize, int seed);

    [[nodiscard]] inline std::vector<int>
    getSources() const { return vertexesNumber; }

    [[nodiscard]] inline std::vector<int>
    getOutEdges(int vertex) const { return v[vertex].next; }

    [[nodiscard]] inline std::vector<int>
    getEdgeNodes(int edge) const { return e[edge].vertexes; }

    [[nodiscard]] inline std::size_t
    nNodes(int edge) const { return e[edge].vertexes.size(); }

    [[nodiscard]] inline std::size_t
    nNodes() const { return v.size(); }

    [[nodiscard]] inline int
    weight(int vertex) const { return v[vertex].weight; }

    void addEdge(std::ifstream &fin);

    void inputWeights(std::ifstream &fin);

    void addLinkedEdges(size_t step);

    void inputRndEdges(int edgeNumber, int edgeSize);

    void inputRndWeights();

    int countCutSet(const std::unordered_map<int, int> *distrib) const;

    void print(const std::unordered_map<int, bool> *sides) const;

    void graphOutput(const std::string &filename,
                     const std::unordered_map<int, bool> *sides) const;

    void dotOutput(std::ofstream &fout,
                   const std::unordered_map<int, bool> *sides) const;
};
