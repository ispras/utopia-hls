#pragma once

#include <cmath>
#include <limits>
#include <map>
#include <utility>
#include <vector>

struct Step {
    int vertex_number, gain, min_rest;
};

struct MultiEdge{
    std::vector<int> vertexes;
    int distrib[2]{};
};

struct Vertex {
    int weight;
    std::vector<int> next;
    bool side = true;
};

void fm(std::vector<Vertex> &v, std::vector<MultiEdge> &e, double r, int passes);
