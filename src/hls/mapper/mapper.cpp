//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/mapper/mapper.h"

#include <cassert>

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

} // namespace eda::hls::mapper
