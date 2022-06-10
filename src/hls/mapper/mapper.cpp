//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/mapper/mapper.h"
#include "util/assert.h"

#include <limits>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#define uassert_node(cond, node, mess) \
  uassert(cond, mess << ": " << node.name << "[" << node.type.name << "]")

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
  uassert_node(!node.map, node, "Node has been already mapped");
  node.map = metaElement;
}

void Mapper::apply(model::Node &node, const model::Parameters &params) {
  uassert_node(node.map, node, "Node has not been mapped");

  // Store the parameters.
  node.params = params;

  // Estimate the node indicators.
  node.map->estimate(params, node.ind);

  // Set the outputs indicators and derive some node indicators.
  node.ind.ticks = node.ind.delay = 0;

  for (auto *output : node.outputs) {
    const auto *port = output->source.port;
    uassert_node(port, node, "Channel has not been linked");

    auto i = node.ind.outputs.find(port->name);
    uassert_node(i != node.ind.outputs.end(), node, "Unspecified output");

    output->ind = i->second;

    node.ind.ticks = std::max(node.ind.ticks, output->ind.ticks);
    node.ind.delay = std::max(node.ind.delay, output->ind.delay);
  }
}

void Mapper::estimate(model::Graph &graph) {
  using V = const Node*;
  using E = std::size_t;
  using D = ChanInd;

  std::stack<std::pair<V, E>> stack;
  std::unordered_set<V> visited;
  std::unordered_map<V, D> distance;

  graph.ind.power = 0;
  graph.ind.area  = 0;
  graph.ind.ticks = 0;
  graph.ind.delay = 0;
  graph.ind.outputs.clear();

  // Iterate over all nodes of the graph.
  for (const auto *node : graph.nodes) {
    // Update the additive characteristics.
    graph.ind.power += node->ind.power;
    graph.ind.area  += node->ind.area;

    // Collect the source nodes for the DFS traveral.
    if (node->isSource()) {
      stack.push({ node, 0 });
      visited.insert(node);
    }
  }

  // Traverse the nodes in topological order (DFS).
  while (!stack.empty()) {
    auto &[node, edge] = stack.top();

    // Visit the node once (when added to the stack).
    if (edge == 0) {
      // Relax all outgoing edges.
      for (const auto *output : node->outputs) {
        const auto *next = output->target.node;

        // Update the distance (indicators) of the adjacent node.
        auto &distNode = distance[node];
        auto &distNext = distance[next];

        const auto newTicks = distNode.ticks + output->ind.ticks;
        const auto newDelay = (output->ind.ticks > 0)
          ? /* New path */   output->ind.delay
          : distNode.delay + output->ind.delay;

        distNext.ticks = std::max(distNext.ticks, newTicks);
        distNext.delay = std::max(distNext.delay, newDelay);

        // Update the integral latency and combinational delay.
        graph.ind.ticks = std::max(graph.ind.ticks, newTicks);
        graph.ind.delay = std::max(graph.ind.delay, newDelay);
      }
    }

    // Schedule the next node (implicit topological sort).
    bool hasMoved = false;

    while (edge < node->outputs.size()) {
      const auto *output = node->outputs[edge++];
      const auto *next = output->target.node;

      // The next node is unvisited.
      if (visited.find(next) == visited.end()) {
        stack.push({ next, 0 });
        visited.insert(next);
        hasMoved = true;
        break;
      }
    }

    // All successors of the node has been traversed.
    if (!hasMoved) {
      stack.pop();
    }
  }
}

void Mapper::estimate(model::Model &model) {
  for (auto *graph : model.graphs) {
    estimate(*graph);
  }

  model.ind = model.main()->ind;
}

} // namespace eda::hls::mapper
