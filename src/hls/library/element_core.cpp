//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_core.h"

namespace eda::hls::library {

std::unique_ptr<Element> ElementCore::construct() const {
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  element->ir = "";
  element->path = path;
  return element;
}

void ElementCore::estimate(const Parameters &params, Indicators &indicators) const {
  // TODO
  indicators.ticks = 1;
  indicators.power = 100;
  indicators.area  = 100;
  indicators.delay = 10;

  ChanInd chanInd;
  chanInd.ticks = indicators.ticks;
  chanInd.delay = indicators.delay;

  indicators.outputs.clear();
  for (const auto &port : ports) {
    if (port.direction != Port::IN) {
      indicators.outputs.insert({ port.name, chanInd });
    }
  }
}

} // namespace eda::hls::library
