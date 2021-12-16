//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "hls/parser/hil/builder.h"

using namespace eda::hls;

namespace eda::hls::parser::hil {

std::shared_ptr<Model> Builder::create() {
  assert(currentModel != nullptr && "No model found");

  for (const NodeType *nodetype: currentModel->nodetypes) {
    assert(nodetype != nullptr);

    // The nodetype allows consuming or producing data.
    uassert(!nodetype->inputs.empty() || !nodetype->outputs.empty(),
      "Nodetype w/o inputs and outputs: " << *nodetype);
  }

  for (const Graph *graph: currentModel->graphs) {
    for (const Chan *chan: graph->chans) {
      assert(chan != nullptr);

      // The channel is attached to nodes.
      uassert(chan->source.isLinked(), "Chan source is not linked: " << *chan);
      uassert(chan->target.isLinked(), "Chan target is not linked: " << *chan);

      // The channel is not a loopback.
      uassert(chan->source.node != chan->target.node,
        "Chan is a self-loop: " << *chan);

      // The source and target datatypes are the same.
      uassert(chan->source.port->type == chan->target.port->type,
        "Chan source and target are of different types: " << *chan);
    }

    for (const Node *node: graph->nodes) {
      assert(node != nullptr);

      // The node corresponds to its type.
      uassert(node->inputs.size() == node->type.inputs.size(),
        "Wrong number of inputs: " << *node);
      uassert(node->outputs.size() == node->type.outputs.size(),
        "Wrong number of outputs: " << *node);
    }
  }

  auto model = std::unique_ptr<Model>(currentModel);
  currentModel = nullptr;

  return model;
}

} // namespace eda::hls::parser::hil
