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
#include "util/singleton.h"

#include <map>
#include <string>

using namespace eda::hls::library;

namespace eda::hls::mapper {

class Mapper final : public Singleton<Mapper> {
  friend class Singleton<Mapper>;

public:
  /// Maps the model nodes to the library meta-elements.
  void map(model::Model &model, Library &library);
  /// Maps the given node to the library meta-element.
  void map(model::Node &node, Library &library);
  /// Maps the given node to the given meta-element.
  void map(model::Node &node, const std::shared_ptr<MetaElement> &metaElement);

  /// Estimates the indicators of the node and the output channels.
  void apply(model::Node &node, const Parameters &params);

private:
  Mapper() {}
};

} // namespace eda::hls::mapper
