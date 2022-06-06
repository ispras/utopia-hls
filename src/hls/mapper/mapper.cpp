//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/mapper/mapper.h"

namespace eda::hls::mapper {

void Mapper::map(Model &model, const Library &library) {
  for (auto *graph : model.graphs) {
    for (auto *node : graph->nodes) {
      // TODO:
    }
  }
}

void Mapper::map(Node &node, const MetaElement &metaElement) {
  // TODO:
}

void Mapper::apply(Node &node, const Parameters &params) {
  // TODO:
}

} // namespace eda::hls::mapper
