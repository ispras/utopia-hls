//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/internal/delay.h"


#include <cmath>

namespace eda::hls::library {

void Delay::estimate(
    const Parameters &params, Indicators &indicators) const {
  unsigned inputCount = 0;
  unsigned latencySum = 0;
  unsigned widthSum = 0;

  unsigned latency = params.getValue(depth);

  unsigned width_value = params.getValue(width);

  for (const auto &port : ports) {
    widthSum += width_value;
    if (port.direction == Port::IN)
      inputCount++;
    else
      latencySum += latency;
  }

  double S = params.getValue(depth);
  double Areg = 1.0;
  double A = S * widthSum * Areg;
  double Fmax = 500000.0;
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

std::shared_ptr<MetaElement> Delay::create(const NodeType &nodetype) {
  std::string name = nodetype.name;
  //If there is no such component in the library then it has to be an internal component.
    std::shared_ptr<MetaElement> metaElement;
    auto ports = createPorts(nodetype);
    std::string lowerCaseName = name;
    unsigned i = 0;
    while (lowerCaseName[i]) {
      lowerCaseName[i] = tolower(lowerCaseName[i]);
      i++;
    }
    Parameters params;
    params.add(Parameter(depth, Constraint(1, std::numeric_limits<unsigned>::max()), 1));
    params.add(Parameter(width, Constraint(1, std::numeric_limits<unsigned>::max()), 1));

    metaElement = std::shared_ptr<MetaElement>(new Delay(lowerCaseName,
                                                         params,
                                                         ports));
  return metaElement;
};

} // namespace eda::hls::library
