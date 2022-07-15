#include "fm.h"

void countDistribution(const std::vector<Vertex> &v, std::vector<MultiEdge> &e) {
    for (auto &edge: e) {
        for (auto &vert: edge.vertexes) {
            ++edge.distrib[v[vert].side];
        }
    }
}

int countGain(const std::vector<Vertex> &v, const std::vector<MultiEdge> &e, size_t i) {
    int fs = 0;
    int ts = 0;
    for (const auto &edge: v[i].next) {
        bool side = v[i].side;
        ts += (e[edge].distrib[side] == static_cast<int>(e[edge].vertexes.size()));
        fs += (e[edge].distrib[side] == 1);
    }
    return fs - ts;
}

int countMinCriterion(const std::vector<Vertex> &v, int lb, int ub, const int *area,
                      size_t i) {
    int narea[]{area[0], area[1]};
    narea[v[i].side] -= v[i].weight;
    narea[!v[i].side] += v[i].weight;
    return std::min(ub - narea[0], narea[0] - lb);
}

int maxGain(const std::vector<Vertex> &v, const std::multimap<int, int, std::greater<>> &gain,
            int lb, int ub, int *area) {
    int cur_gain = gain.begin()->first;
    int max_vertex = -1;
    int max_rest = 0;
    for (const auto &[gain_val, vertex]: gain) {
        if (gain_val < cur_gain && max_vertex != -1) {
            return max_vertex;
        }
        int cur_rest = countMinCriterion(v, lb, ub, area, vertex);
        if (cur_rest >= max_rest) {
            cur_gain = gain_val;
            max_vertex = vertex;
            max_rest = cur_rest;
        }
    }
    return max_vertex;
}

void moveVertex(int vertex, std::vector<Vertex> &v, std::vector<MultiEdge> &e, int *area) {
    bool side = v[vertex].side;
    area[side] -= v[vertex].weight;
    area[!side] += v[vertex].weight;
    v[vertex].side = !side;
    for (auto &edge: v[vertex].next) {
        --e[edge].distrib[side];
        ++e[edge].distrib[!side];
    }
}

void tempMove(int vertex, std::vector<Vertex> &v, std::vector<MultiEdge> &e, std::vector<Step> &order,
              std::multimap<int, int, std::greater<>> &gain,
              std::vector<std::multimap<int, int>::iterator> &ptr_gain,
              int *area, int lb, int ub) {
    order.push_back({vertex, ptr_gain[vertex]->first, countMinCriterion(v, lb, ub, area, vertex)});
    gain.erase(ptr_gain[vertex]);
    ptr_gain[vertex] = gain.end();
    moveVertex(vertex, v, e, area);
}

void randPartition(std::vector<Vertex> &v, int area[2], int lb, int ub) {
    int middle = (ub - lb) / 2 + lb;
    std::vector<std::vector<int>> sum(v.size() + 1, std::vector<int>(middle + 1));
    for (int wei = 1; wei <= middle; ++wei) {
        for (size_t vert = 1; vert <= v.size(); ++vert) {
            int add = sum[vert - 1][wei];
            if (wei >= v[vert - 1].weight) {
                sum[vert][wei] =
                        std::max(add, sum[vert - 1][wei - v[vert - 1].weight] + v[vert - 1].weight);
            } else {
                sum[vert][wei] = add;
            }
        }
    }
    int vert = static_cast<int>(v.size());
    int wei = middle;
    while (sum[vert][wei]) {
        if (sum[vert][wei] != sum[vert - 1][wei]) {
            v[vert - 1].side = !v[vert - 1].side;
            wei -= v[vert - 1].weight;
        }
        --vert;
    }
    area[0] = area[1] = 0;
    for (Vertex &vertex: v) {
        area[vertex.side] += vertex.weight;
    }
}

void balanceCriterion(std::vector<Vertex> &v, double r, int *lb, int *ub) {
    int max_area = 0;
    int area = 0;
    for (auto &vertex: v) {
        if (max_area < vertex.weight) {
            max_area = vertex.weight;
        }
        area += vertex.weight;
    }
    *lb = static_cast<int>(std::ceil(r * area)) - max_area;
    *ub = static_cast<int>(r * area) + max_area;
}

void gainUpdate(std::vector<Vertex> &v, const std::vector<MultiEdge> &e, size_t moved_vertex,
                std::multimap<int, int, std::greater<>> &gain,
                std::vector<std::multimap<int, int>::iterator> &ptr_gain) {
    for (auto &edge: v[moved_vertex].next) {
        bool to_block = v[moved_vertex].side;
        int tf[2] = {e[edge].distrib[to_block] - 1, e[edge].distrib[!to_block]};
        if (tf[0] <= 1 || tf[1] <= 1) {
            for (const auto &vertex: e[edge].vertexes) {
                if (ptr_gain[vertex] != gain.end()) {
                    int inc = 0;
                    if (tf[0] == 0) {
                        ++inc;
                    } else if (tf[0] == 1 && v[vertex].side == to_block) {
                        --inc;
                    }
                    if (tf[1] == 0) {
                        --inc;
                    } else if (tf[1] == 1 && v[vertex].side != to_block) {
                        ++inc;
                    }
                    int new_gain = ptr_gain[vertex]->first + inc;
                    gain.erase(ptr_gain[vertex]);
                    ptr_gain[vertex] = gain.emplace(new_gain, vertex);
                }
            }
        }
    }
}

void bestMoves(const std::vector<Step> &order, int &Gm, int &m) {
    Gm = std::numeric_limits<int>::min();
    m = 0;
    int sum = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        sum += order[i].gain;
        if (sum > Gm || (sum == Gm && order[i].min_rest > order[m].min_rest)) {
            Gm = sum;
            m = static_cast<int>(i);
        }
    }
}

void confirmMoves(std::vector<Vertex> &v, std::vector<MultiEdge> &e, int *area, const std::vector<Step> &order, int m) {
    for (int i = static_cast<int>(order.size()) - 1; i != m; --i) {
        int vertex = order[i].vertex_number;
        moveVertex(vertex, v, e, area);
    }
}

void fm(std::vector<Vertex> &v, std::vector<MultiEdge> &e, double r, int passes) {
    int lower, upper; // lower bound, upper bound.
    balanceCriterion(v, r, &lower, &upper);
    int area[2];
    randPartition(v, area, lower, upper);
    countDistribution(v, e);
    int m, Gm = std::numeric_limits<int>::max();
    int iteration = 0;
    while (Gm > 0 && iteration < passes) {
        ++iteration;
        std::vector<Step> order;
        order.reserve(v.size());
        size_t fixed_numb = 0;
        std::multimap<int, int, std::greater<>> gain; // gain - vertex number.
        std::vector<std::multimap<int, int, std::greater<>>::iterator> ptr_gain(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            ptr_gain[i] = gain.emplace(countGain(v, e, i), static_cast<int>(i));
        }
        while (fixed_numb < v.size()) {
            int vertex_number = maxGain(v, gain, lower, upper, area);
            if (vertex_number < 0) {
                break;
            }
            tempMove(vertex_number, v, e, order, gain, ptr_gain, area, lower, upper);
            ++fixed_numb;
            gainUpdate(v, e, vertex_number, gain, ptr_gain);
        }
        bestMoves(order, Gm, m);
        if (Gm > 0) {
            confirmMoves(v, e, area, order, m);
        }
    }
}
