//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_generator.h"

#include <cmath>

namespace eda::hls::library {

std::unique_ptr<Element> ElementGenerator::construct(
    const Parameters &params) const {
  system((genPath + " " +
          "." + " " +
          "mul" + " " +
          "16").c_str());
  std::unique_ptr<Element> element = std::make_unique<Element>(ports);
  element->ir = "";
  element->path = "./mul.v";
  return element;
}


void ElementGenerator::estimate(
    const Parameters &params, Indicators &indicators) const {
  unsigned inputCount = 0;
  unsigned latencySum = 0;
  unsigned widthSum = 0;

  for (const auto &port : ports) {
    widthSum+=port.width;
    if (port.direction == Port::IN)
      inputCount++;
    else
      latencySum += port.latency;
  }

  unsigned S = params.value("stages");
  double Areg = 1.0;
  double Apipe = S * widthSum * Areg;
  double Fmax = 300.0;
  double F = Fmax * (1 - std::exp(0.5 - S));
  double C = inputCount * latencySum;
  double N = (C == 0 ? 0 : C * std::log((Fmax / (Fmax - F)) * ((C - 1) / C)));
  double A = C * std::sqrt(N) + Apipe;
  double P = A;

  indicators.frequency  = static_cast<unsigned>(F);
  indicators.throughput = static_cast<unsigned>(F);
  indicators.latency    = static_cast<unsigned>(N);
  indicators.power      = static_cast<unsigned>(P);
  indicators.area       = static_cast<unsigned>(A);
}

} // namespace eda::hls::library
