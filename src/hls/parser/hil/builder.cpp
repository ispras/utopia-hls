//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <memory>

#include "hls/parser/hil/builder.h"

using namespace eda::hls;

namespace eda::hls::parser::hil {

std::unique_ptr<Builder> Builder::_instance = nullptr;

std::unique_ptr<Model> Builder::create() {
  CHECK(_model != nullptr, "Model is null");

  for (const NodeType *nodetype: _model->nodetypes) {
    CHECK(nodetype != nullptr, "NodeType is null");

    // The nodetype allows consuming or producing data.
    CHECK(!nodetype->inputs.empty() || !nodetype->outputs.empty(),
      "NodeType w/o inputs and outputs: " << *nodetype);
  }

  for (const Graph *graph: _model->graphs) {
    for (const Chan *chan: graph->chans) {
      CHECK(chan != nullptr, "Chan is null");

      // The channel is attached to nodes.
      CHECK(chan->source.is_linked(), "Chan source is not linked: " << *chan);
      CHECK(chan->target.is_linked(), "Chan target is not linked: " << *chan);

      // The channel is not a loopback.
      CHECK(chan->source.node != chan->target.node,
        "Chan is a self-loop: " << chan);

      // The source and target datatypes are the same.
      CHECK(chan->source.port->type == chan->target.port->type,
        "Chan source and target are of different types: " << *chan);
    }

    for (const Node *node: graph->nodes) {
      CHECK(node != nullptr, "Node is null");

      // The node corresponds to its type.
      CHECK(node->inputs.size() == node->type.inputs.size(),
        "Wrong number of inputs: " << *node);
      CHECK(node->outputs.size() == node->type.outputs.size(),
        "Wrong number of outputs: " << *node);
    }
  }

  auto model = std::unique_ptr<Model>(_model);
  _model = nullptr;

  return model;
}

} // namespace eda::hls::parser::hil
