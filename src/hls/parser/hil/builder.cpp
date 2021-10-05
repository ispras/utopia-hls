//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <memory>

#include "hls/parser/hil/builder.h"

using namespace eda::hls;

namespace eda::hls::parser::hil {

std::unique_ptr<Builder> Builder::_instance = nullptr;

std::unique_ptr<Model> Builder::create() {
  assert(_model != nullptr);

  for (const NodeType *nodetype: _model->nodetypes) {
    assert(nodetype != nullptr);

    // The nodetype allows consuming or producing data.
    assert(!nodetype->inputs.empty() || !nodetype->outputs.empty());
  }

  for (const Graph *graph: _model->graphs) {
    for (const Chan *chan: graph->chans) {
      assert(chan != nullptr);

      // The channel is attached to nodes.
      assert(chan->source != nullptr);
      assert(chan->target != nullptr);

      // The channel is not a loopback.
      assert(chan->source != chan->target);

      // The source and target datatypes are the same.
      assert(chan->source->type == chan->target->type);
    }

    for (const Node *node: graph->nodes) {
      assert(node != nullptr);

      // The node corresponds to its type.
      assert(node->inputs.size() == node->type.inputs.size());
      assert(node->outputs.size() == node->type.outputs.size());
    }
  }

  auto model = std::unique_ptr<Model>(_model);
  _model = nullptr;

  return model;
}

} // namespace eda::hls::parser::hil
