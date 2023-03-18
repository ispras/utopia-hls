//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"

namespace eda::hls::library::internal {

std::vector<Port> ElementInternal::createPorts(const NodeType &nodetype) {
  std::vector<Port> ports;

  ports.push_back(Port("clock",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1),
                       Port::Type::CLOCK));

  ports.push_back(Port("reset",
                       Port::IN,
                       1,
                       model::Parameter(std::string("width"), 1),
                       Port::Type::RESET));

  for (const auto *input: nodetype.inputs) {
    ports.push_back(Port(input->name,
                         Port::IN,
                         input->type.size,
                         model::Parameter(std::string("width"),
                                          input->type.size)));
  }

  for (const auto *output: nodetype.outputs) {
    ports.push_back(Port(output->name,
                         Port::OUT,
                         output->type.size,
                         model::Parameter(std::string("width"),
                                          output->type.size)));
  }
  return ports;
}

} // namespace eda::hls::library::internal