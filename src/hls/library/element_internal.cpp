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
                         1,
                         model::Parameter(std::string("width"), 16)));
  }

  for (const auto *input: nodetype.outputs) {
    ports.push_back(Port(input->name,
                         Port::OUT,
                         1,
                         model::Parameter(std::string("width"), 16)));
  }
  return ports;
}

std::shared_ptr<MetaElement> ElementInternal::create(const NodeType &nodetype) {
  std::string name = nodetype.name;
  //If there is no such component in the library then it has to be an internal component.
  std::shared_ptr<MetaElement> metaElement;
  if (nodetype.isDelay()) {
    metaElement = Delay::create(nodetype);
  } else {
    metaElement = Default::create(nodetype);
  }
  return metaElement;
}

} // namespace eda::hls::library
