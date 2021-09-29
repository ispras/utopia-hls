/*
 * Copyright 2021 ISP RAS (http://www.ispras.ru)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

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
