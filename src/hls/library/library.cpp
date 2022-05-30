//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include "hls/library/ipxact_parser.h"
#include "hls/library/library.h"
#include "hls/library/library_mock.h"
#include "util/assert.h"

#include <algorithm>

namespace eda::hls::library {

void Library::initialize(const std::string &config) {
  IPXACTParser::get().initialize();
  IPXACTParser::get().parseCatalog(config);
}

void Library::finalize() {
  IPXACTParser::get().finalize();
}

std::shared_ptr<MetaElement> Library::find(const NodeType &nodetype) {
  return find(nodetype.name);
}

std::shared_ptr<MetaElement> Library::find(const std::string &name) {
  const auto i = std::find_if(cache.begin(), cache.end(),
    [&name](const std::shared_ptr<MetaElement> &metaElement) {
      return metaElement->name == name;
    });

  if (i == cache.end()) {
    auto metaElement = IPXACTParser::get().parseComponent(name);
    cache.push_back(metaElement);
    return metaElement;
  }

  return *i;
}

} // namespace eda::hls::library
