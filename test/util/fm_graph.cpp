//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "fm_graph.h"

Graph::Graph(size_t nodesSize) {
    v.resize(nodesSize);
    vertexesNumber.resize(nodesSize);
    for (size_t i = 0; i < nodesSize; ++i) {
        vertexesNumber[i] = static_cast<int>(i);
    }
}

Graph::Graph(size_t nodesSize, int seed) : Graph(nodesSize) {
    std::srand(seed);
}

void Graph::inputWeights(std::ifstream &fin) {
    for (auto &vertex: v) {
        fin >> vertex.weight;
    }
}

void Graph::addEdge(std::ifstream &fin) {
    int vert_number;
    fin >> vert_number;
    e.emplace_back(MultiEdge{std::vector<int>(vert_number)});
    for (int &vertex: e.back().vertexes) {
        fin >> vertex;
        v[vertex].next.push_back(static_cast<int>(e.size()) - 1);
    }
}

void Graph::inputRndEdges(int edgNumber, int edgSize) {
    e.resize(edgNumber);
    std::vector<int> limits(edgNumber);
    for (size_t i = 0; i < v.size(); ++i) {
        Vertex &vert = v[i];
        int edge = rand() % edgNumber;
        if (limits[edge] >= edgSize) {
            edge = 0;
        }
        vert.next.push_back(edge);
        e[edge].vertexes.push_back(static_cast<int>(i));
        ++limits[edge];
    }
    for (size_t i = 0; i < limits.size(); ++i) {
        if (limits[i] == 0) {
            int ver = rand() % (v.size() - 1);
            e[i].vertexes.push_back(ver);
            e[i].vertexes.push_back(ver + 1);
            v[ver].next.push_back(i);
            v[ver + 1].next.push_back(i);
        } else if (limits[i] == 1) {
            int ver = (e[i].vertexes[0] + 1) % static_cast<int>(v.size());
            e[i].vertexes.push_back(ver);
            v[ver].next.push_back(i);
        }
    }
}

void Graph::inputRndWeights() {
    for (auto &ver: v) {
        ver.weight = rand() % 100 + 1;
    }
}

void Graph::addLinkedEdges(size_t step) {
    e = std::vector<MultiEdge>(
            (v.size() % (step - 1) != 0) + v.size() / (step - 1),
            MultiEdge{std::vector<int>(step)});
    int edge = 0;
    for (size_t i = 0; i < v.size(); i += (step - 1)) {
        for (size_t j = 0; j < step; ++j) {
            e[edge].vertexes[j] = static_cast<int>((j + i) % v.size());
        }
        for (int vertex: e[edge].vertexes) {
            v[vertex].next.push_back(edge);
        }
        ++edge;
    }
}

int Graph::countCutSet(const std::unordered_map<int, int> *distrib) const {
    int cutset = 0;
    for (size_t i = 0; i < e.size(); ++i) {
        auto itSide1 = distrib[0].find(static_cast<int>(i));
        auto itSide2 = distrib[1].find(static_cast<int>(i));
        cutset += itSide1 != distrib[0].end() && itSide1->second != 0 &&
                  itSide2 != distrib[1].end() && itSide2->second != 0;
    }
    return cutset;
}

void Graph::print(const std::unordered_map<int, bool> *sides) const {
    int area[2]{};
    for (int side = 0; side < 2; ++side) {
        std::cout << side << " : {";
        for (size_t i = 0; i < v.size(); ++i) {
            if (sides->at(static_cast<int>(i)) == side) {
                area[side] += v[i].weight;
                std::cout << " " << i;
            }
        }
        std::cout << " } ";
        std::cout << area[side] << '\n';
    }
    std::cout << std::endl;
}

void Graph::graphOutput(const std::string &filename,
                        const std::unordered_map<int, bool> *sides) const {
    std::ofstream fout(filename);
    if (fout.is_open()) {
        dotOutput(fout, sides);
        fout.close();
    }
}

void Graph::dotOutput(std::ofstream &fout,
                      const std::unordered_map<int, bool> *sides) const {
    const char *colors[] = {"blue", "red"};
    fout << "graph partitioned {\n";
    for (int side = 0; side < 2; ++side) {
        fout << "\tsubgraph cluster_" << side << " {\n";
        for (size_t i = 0; i < v.size(); ++i) {
            if (sides->at(i) == side) {
                fout << "\t\tv" << i;
                fout << ";\n";
            }
        }
        for (size_t i = 0; i < e.size(); ++i) {
            bool there = false;
            for (int vert: e[i].vertexes) {
                there |= (sides->at(vert) != side);
            }
            if (!there) {
                fout << "\t\te" << i << "[shape=point];\n";
            }
        }
        fout << "\t\tcolor=" << colors[side] << ";\n";
        fout << "\t}\n";
    }
    for (size_t i = 0; i < e.size(); ++i) {
        fout << "\te" << i << "[shape=point];\n";
        for (int vert: e[i].vertexes) {
            if (vert & 1) {
                fout << "\te" << i << " -- v" << vert << ";\n";
            } else {
                fout << "\tv" << vert << " -- e" << i << ";\n";
            }
        }
    }
    fout << "}";
}
