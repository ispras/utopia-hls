//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_generator.h"

namespace eda::hls::library {

/// TODO: Realize TGI
std::unique_ptr<Element> ElementGenerator::construct() const {
  system((genPath +
              " " +
              "." +
              " " +
          "mul.v" +
              " " +
              "16").c_str());
  auto element = std::make_unique<Element>(ports);
  element->ir = "";
  element->path = "./mul.v";
  return element;
}

void ElementGenerator::estimate(const Parameters &params,
                                Indicators &indicators) const {
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