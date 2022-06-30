//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/internal/default.h"
#include "hls/library/internal/delay.h"
#include "hls/library/internal/dup.h"


#include <cmath>

namespace eda::hls::library {

std::vector<Port> ElementInternal::createPorts(const NodeType &nodetype) {
  std::vector<Port> ports;

  ports.push_back(Port("clock",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1)));

  ports.push_back(Port("reset",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1)));

  for (const auto *input: nodetype.inputs) {
    ports.push_back(Port(input->name,
                         Port::IN,
                         16,
                         model::Parameter(std::string("width"), 16)));
  }

  for (const auto *output: nodetype.outputs) {
    ports.push_back(Port(output->name,
                         Port::OUT,
                         16,
                         model::Parameter(std::string("width"), 16)));
  }
  return ports;
}

std::shared_ptr<MetaElement> ElementInternal::create(const NodeType &nodetype,
                                                     const HWConfig &hwconfig) {
  std::string name = nodetype.name;
  std::shared_ptr<MetaElement> metaElement;
  if (nodetype.isDelay()) {
    metaElement = Delay::create(nodetype, hwconfig);
  } else if (nodetype.isDup()) {
    metaElement = Dup::create(nodetype, hwconfig);
  } else {
    metaElement = Default::create(nodetype, hwconfig);
  }
  return metaElement;
}

} // namespace eda::hls::library
