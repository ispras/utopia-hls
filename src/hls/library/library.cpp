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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <string.h>

namespace eda::hls::library {

void Library::initialize(const std::string &libPath,
                         const std::string &catalogPath) {
  IPXACTParser::get().initialize();
  IPXACTParser::get().parseCatalog(libPath, catalogPath);
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

void MetaElement::callGen() const {
  //FIXME
  if (hasGen) {
  system((genPath + " " +
          "/home/grigorovia/utopia" + " " +
          "mul" + " " +
          "32").c_str());
  } else {
    std::cout << "Component is not a generator!" << std::endl;
  }
}

} // namespace eda::hls::library
