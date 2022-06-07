//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/element_internal.h"
#include "hls/library/ipxact_parser.h"
#include "hls/library/library.h"
#include "util/assert.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

namespace eda::hls::library {

void Library::initialize(const std::string &libraryPath,
                         const std::string &catalogPath) {
  IPXACTParser::get().initialize();
  IPXACTParser::get().parseCatalog(libraryPath, catalogPath);
}

void Library::finalize() {
  IPXACTParser::get().finalize();
}

std::shared_ptr<MetaElement> Library::find(const NodeType &nodetype) {
  const auto name = nodetype.name;

  const auto i = std::find_if(cache.begin(), cache.end(),
    [&name](const std::shared_ptr<MetaElement> &metaElement) {
      return metaElement->name == name;
    });

  if (i != cache.end()) {
    return *i;
  }

  if (IPXACTParser::get().hasComponent(name)) {
    auto metaElement = IPXACTParser::get().parseComponent(name);
    cache.push_back(metaElement);
    return metaElement;
  }
  auto metaElement = create(nodetype);
  return metaElement;
}


std::shared_ptr<MetaElement> Library::create(const NodeType &nodetype) {
  std::string name = nodetype.name;
  //If there is no such component in the library then it has to be an internal component.
  Parameters params;
  params.add(Parameter("f", Constraint(1, 1000), 100));
  params.add(Parameter("stages", Constraint(0, 10000), 10));

  std::vector<Port> ports;

  for (const auto *input: nodetype.inputs) {
    ports.push_back(Port(input->name, Port::IN, input->latency, 16, false, ' '));//TODO
  }
  for (const auto *output: nodetype.outputs) {
    ports.push_back(Port(output->name, Port::OUT, output->latency, 16, false, ' '));//TODO
  }

  // Add clk and rst ports: these ports are absent in the lists above.
  ports.push_back(Port("clock", Port::IN, 0, 1, false, ' '));
  ports.push_back(Port("reset", Port::IN, 0, 1, false, ' '));

  std::string lowerCaseName = name;
  unsigned i = 0;
  while (lowerCaseName[i]) {
    lowerCaseName[i] = tolower(lowerCaseName[i]);
    i++;
  }
  return std::shared_ptr<MetaElement>(new ElementInternal(lowerCaseName,
                                                          params,
                                                          ports));
}

} // namespace eda::hls::library
