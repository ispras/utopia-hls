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

void Mapper::map(Model &model, Library &library) {
  for (auto *graph : model.graphs) {
    for (auto *node : graph->nodes) {
      map(*node, library);
    }
  }
}

void Mapper::map(Node &node, Library &library) {
  auto metaElement = library.find(node.type);
  assert(metaElement != nullptr);
  map(node, metaElement);
}

void Mapper::map(Node &node, const std::shared_ptr<MetaElement> &metaElement) {
  assert(!node.map && "Node has been already mapped");
  node.map = metaElement;
}

void Mapper::apply(Node &node, const Parameters &params) {
  assert(node.map && "Node is unmapped");

  
}

} // namespace eda::hls::mapper
