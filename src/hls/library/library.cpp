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
#include "hls/mapper/config/hwconfig.h"
#include "util/assert.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_map>

using namespace eda::hls::mapper;

namespace eda::hls::library {

bool MetaElement::supports(const HWConfig &hwconfig) {
  return true;
}

void Library::initialize(const std::string &libraryPath,
                         const std::string &catalogPath) {
  IPXACTParser::get().initialize();
  IPXACTParser::get().parseCatalog(libraryPath, catalogPath);
}

void Library::finalize() {
  IPXACTParser::get().finalize();
}

void Library::importLibrary(const std::string &libraryPath,
                            const std::string &catalogPath) {
  IPXACTParser::get().parseCatalog(libraryPath, catalogPath);
}

std::shared_ptr<MetaElement> Library::find(const NodeType &nodetype,
                                           const HWConfig &hwconfig) {
  std::string hashString = nodetype.name;

  for (const auto *input: nodetype.inputs) {
    hashString = hashString + " in " + input->name;
  }

  for (const auto *output: nodetype.outputs) {
    hashString = hashString + " out " + output->name;
  }

  /*const auto i = std::find_if(cache.begin(), cache.end(),
    [&name, &hwconfig](const std::shared_ptr<MetaElement> &metaElement) {
      return metaElement->name == name && metaElement->supports(hwconfig);
    });

  if (i != cache.end()) {
    return *i;
  }*/

  const auto hash = std::hash<std::string>{}(hashString);

  if (cache.count(hash) > 0) {
    return cache[hash];
  }

  if (IPXACTParser::get().hasComponent(nodetype.name, hwconfig)) {
    auto metaElement = IPXACTParser::get().parseComponent(nodetype.name);
    cache.insert({hash, metaElement});
    return metaElement;
  }
  auto metaElement = ElementInternal::create(nodetype, hwconfig);
  cache.insert({hash, metaElement});
  return metaElement;
}

} // namespace eda::hls::library
