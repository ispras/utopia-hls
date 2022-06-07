//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/mapper/mapper.h"

#include <cassert>
#include <limits>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace eda::hls::mapper {

void Mapper::map(model::Model &model, Library &library) {
  for (auto *graph : model.graphs) {
    for (auto *node : graph->nodes) {
      map(*node, library);
    }
  }
}

void Mapper::map(model::Node &node, Library &library) {
  auto metaElement = library.find(node.type);
  assert(metaElement != nullptr);
  map(node, metaElement);
}

void Mapper::map(model::Node &node, const std::shared_ptr<MetaElement> &metaElement) {
  assert(!node.map && "Node has been already mapped");
  node.map = metaElement;
}

void Mapper::apply(model::Node &node, const Parameters &params) {
  assert(node.map && "Node is unmapped");

  // Estimate the node indicators.
  node.map->estimate(params, node.ind);

  // Set the outputs indicators and derive some node indicators.
  node.ind.latency = node.ind.delay = 0;

  for (auto *output : node.outputs) {
    const auto *port = output->source.port;
    assert(port && "Channel is unlinked");

    auto i = node.ind.outputs.find(port->name);
    assert(i != node.ind.outputs.end() && "Unspecified output");

    output->ind = i->second;

    node.ind.latency = std::max(node.ind.latency, output->ind.latency);
    node.ind.delay = std::max(node.ind.delay, output->ind.delay);
  }
}

void Mapper::estimate(model::Graph &graph) {
  graph.ind.power   = 0;
  graph.ind.area    = 0;
  graph.ind.latency = 0;
  graph.ind.delay   = 0;
  graph.ind.outputs.clear();

  std::stack<std::pair<const Node*, std::size_t>> stack;
  std::unordered_set<const Node*> visited;
  std::unordered_map<const Node*, unsigned> dist;

  for (auto *node : graph.nodes) {
    graph.ind.power += node->ind.power;
    graph.ind.area  += node->ind.area;

    // FIXME:
    graph.ind.delay = std::max(graph.ind.delay, node->ind.delay);

    if (node->isSource()) {
      stack.push({ node, 0 });
      visited.insert(node);
    }
  }

  // Traverse the nodes in topological order (DFS).
  while (!stack.empty()) {
    auto &[node, i] = stack.top();

    // Visit the node.
    if (i == 0) {
      for (auto *output : node->outputs) {
        const auto *next = output->target.node;

        // Update the distance (relax the edge).
        auto &Dold = dist[next];
        auto  Dnew = dist[node] + output->ind.latency;
        Dold = std::max(Dold, Dnew);

        // Update the integral latency.
        graph.ind.latency = std::max(graph.ind.latency, Dnew);
      }
    }

    // Schedule the next node (implicit topological sort).
    bool hasMoved = false;

    while (i < node->outputs.size()) {
      const auto *output = node->outputs[i++];
      const auto *next = output->target.node;

      if (visited.find(next) == visited.end()) {
        stack.push({ next, 0 });
        visited.insert(next);
        hasMoved = true;
        break;
      }
    }

    if (!hasMoved) {
      stack.pop();
    }
  }
}

void Mapper::estimate(model::Model &model) {
  for (auto *graph : model.graphs) {
    estimate(*graph);

    if (graph->isMain()) {
      model.ind = graph->ind;
    }
  }
}

} // namespace eda::hls::mapper
