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

using namespace eda::hls::model;
using namespace eda::hls::library;

namespace eda::hls::mapper {

/// Node indicators.
struct NodeInd {
  /// Frequency (Hz).
  unsigned freq;
  /// Power (mW).
  unsigned power;
  /// Area (cells).
  unsigned area;
};

/// Chan indicators.
struct ChanInd {
  /// Latency (ticks).
  unsigned latency;
  /// Combinational delay (ns).
  unsigned delay;
};

class Mapper final : public Singleton<Mapper> {
  friend class Singleton<Mapper>;

public:
  /// Maps the model nodes to the library meta-elements.
  void map(Model &model, const Library &library);
  /// Maps the given node to the given meta-element.
  void map(Node &node, const MetaElement &metaElement);

  /// Estimates the node and chan indicators.
  void apply(Node &node, const Parameters &params);

private:
  Mapper() {}
};

} // namespace eda::hls::mapper
