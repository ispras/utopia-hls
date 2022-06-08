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
  /*std::cout << "***********************************************" << std::endl;
  std::cout << genPath << std::endl;
  std::cout << "***********************************************" << std::endl;*/
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

  unsigned S = params.getValue("stages");
  double Areg = 1.0;
  double Apipe = S * widthSum * Areg;
  double Fmax = 300.0;
  double F = Fmax * (1 - std::exp(0.5 - S));
  double C = inputCount * latencySum;
  double N = (C == 0 ? 0 : C * std::log((Fmax / (Fmax - F)) * ((C - 1) / C)));
  double A = C * std::sqrt(N) + Apipe;
  double P = A;
  double D = 1000000000.0/Fmax;

  indicators.ticks = static_cast<unsigned>(N);
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
