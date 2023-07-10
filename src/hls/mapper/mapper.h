//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "hls/model/model.h"
#include "utils/singleton.h"

#include <map>
#include <string>

namespace eda::hls::mapper {

class Mapper final : public Singleton<Mapper> {
  friend class Singleton<Mapper>;

public:
  /// Maps the model nodes to the library meta-elements.
  void map(model::Model &model, library::Library &library);
  /// Maps the given node to the library meta-element.
  void map(model::Node &node, library::Library &library);
  /// Maps the given node to the given meta-element.
  void map(model::Node &node,
           const std::shared_ptr<library::MetaElement> &metaElement);

  /// Applies the given parameters to the given node.
  void apply(model::Node &node, const model::Parameters &params);

  /// Estimates the model indicators.
  void estimate(model::Graph &graph);
  /// Estimates the graph indicators.
  void estimate(model::Model &model);

private:

  Mapper() {}
};

} // namespace eda::hls::mapper