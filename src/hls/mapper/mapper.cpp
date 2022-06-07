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

void Mapper::estimate(const model::Model &model, Library &library,
    const std::map<std::string, Parameters> &params, Indicators &indicators) const {

  Graph *graph = model.main();
  const unsigned maxFrequency = 1000000;

  // Estimate the integral indicators.
  indicators.freq = maxFrequency;
  indicators.power = 0;
  indicators.area = 0;

  for (const auto *node : graph->nodes) {
    auto metaElement = library.find(node->type);
    assert(metaElement && "MetaElement is not found");
    auto nodeParams = params.find(node->name);

    Indicators nodeIndicators;

    Parameters *tempParams = nullptr;
    if (nodeParams == params.end()) {
      // FIXME
      Constraint constr(1000, 500000);
      tempParams = new Parameters(metaElement->params);
      tempParams->add(Parameter("f", constr, constr.max)); // FIXME
      tempParams->add(Parameter("stages", Constraint(0, 10000), 10));  // FIXME
    }

    metaElement->estimate(
      nodeParams == params.end()
        ? *tempParams         // For inserted nodes
        : nodeParams->second, // For existing nodes
      nodeIndicators);

    indicators.power += nodeIndicators.power;
    indicators.area += nodeIndicators.area;

    // Set the minimal frequency rate
    if (nodeIndicators.freq < indicators.freq) {
      indicators.freq = nodeIndicators.freq;
    }
  }
  indicators.perf = indicators.freq;
}

} // namespace eda::hls::mapper
