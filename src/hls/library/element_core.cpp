//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#include "hls/library/element_core.h"

#include <cmath>

namespace eda::hls::library {

std::unique_ptr<Element> ElementCore::construct(
      const Parameters &params) const {
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  element->ir = "";
  element->path = path;
  return element;
}

void ElementCore::estimate(
    const Parameters &params, Indicators &indicators) const {
  unsigned widthSum = 0;

  for (const auto &port : ports) {
    widthSum+=port.width;
  }

  unsigned S = params.getValue("stages");
  double Areg = 1.0;
  double Apipe = S * widthSum * Areg;
  double Fmax = 300.0;
  double F = Fmax * (1 - std::exp(0.5 - S));
  double C = widthSum;
  double N = (C == 0 ? 0 : C * std::log((Fmax / (Fmax - F)) * ((C - 1) / C)));
  double A = C * std::sqrt(N) + Apipe;
  double P = A;
  double D = 1000000000.0 / Fmax;

  indicators.ticks = static_cast<unsigned>(S);
  indicators.power = static_cast<unsigned>(P);
  indicators.area  = static_cast<unsigned>(A);
  indicators.delay = static_cast<unsigned>(D);

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
