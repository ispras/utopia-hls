//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "hls/parser/hil/builder.h"

using namespace eda::hls;

namespace eda::hls::parser::hil {

std::shared_ptr<Model> Builder::create() {
  uassert(currentModel != nullptr, "No model found!\n");

  // Check nodetypes.
  for (auto nodeTypeIterator = currentModel->nodetypes.begin();
       nodeTypeIterator != currentModel->nodetypes.end();
       nodeTypeIterator++) {
    const auto *nodetype = nodeTypeIterator->second;
    uassert(nodetype != nullptr, "NodeType is nullptr!\n");

    // The nodetype allows consuming or producing data.
    uassert(!nodetype->inputs.empty() || !nodetype->outputs.empty(),
      "Nodetype w/o inputs and outputs: " << *nodetype << "\n!");
  }

  // Check graphs.
  for (const Graph *graph: currentModel->graphs) {
    // Check chans.
    for (const Chan *chan: graph->chans) {
      uassert(chan != nullptr, "Chan is nullptr!\n" << "\n!");

      // The channel is attached to nodes.
      uassert(chan->source.isLinked(), "Chan source is not linked: " << *chan
              << "!\n");
      uassert(chan->target.isLinked(), "Chan target is not linked: " << *chan
              << "!\n");

      // The channel is not a loopback.
      uassert(chan->source.node != chan->target.node,
        "Chan is a self-loop: " << *chan << "\n!");

      // The source and target datatypes are the same.
      uassert(chan->source.port->type.name == chan->target.port->type.name,
          "Chan source and target are of different types: " << *chan << "\n!");

    }

    // Check nodes.
    for (const Node *node: graph->nodes) {
      uassert(node != nullptr, "Node is nullptr!\n");

      // The node corresponds to its type.
      uassert(node->inputs.size() == node->type.inputs.size(),
        "Wrong number of inputs: " << *node << "\n!");
      uassert(node->outputs.size() == node->type.outputs.size(),
        "Wrong number of outputs: " << *node << "\n!");
    }

    // Check connections.
    for (const Con *con: graph->cons) {
      uassert(con != nullptr, "Con is nullptr!\n");

      // The connection is attached to nodes.
      uassert(con->source.isLinked(), "Con source is not linked: " << *con
              << "!\n");
      uassert(con->target.isLinked(), "Con target is not linked: " << *con
              << "!\n");

      // The connection is not a loopback.
      uassert(con->source.node != con->target.node,
        "Con is a self-loop: " << *con << "\n!");

       // The source and target datatypes are the same.
      uassert(con->source.port->type.name == con->target.port->type.name,
          "Con source and target are of different types: " << *con << "\n!");

      // The connection must be a link between different graphs.
      uassert(con->source.graph->name != con->target.graph->name,
              "Con source graph and target graph are the same!\n");
    }
  }

  auto model = std::unique_ptr<Model>(currentModel);
  currentModel = nullptr;

  return model;
}

} // namespace eda::hls::parser::hil